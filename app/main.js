// gutfeeling desktop shell.
// Ctrl+Shift+Space -> screenshot -> snip window: splash card over the screenshot while the models load, then the
// screen darkens into the snip view -> results.
const { app, BrowserWindow, globalShortcut, desktopCapturer, screen, ipcMain, Tray, Menu, nativeImage, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const HOTKEY = 'CommandOrControl+Shift+Space';
const REPO = path.resolve(__dirname, '..');
// six outside-lab fields (3 SSL, 3 HP; SPIDER test split) with a 100 um bar, built by research/make_outside_demo.py;
// opened at start-up only with DEMHIST_DEMO=1
const DEMO_IMAGE = path.join(REPO, 'data', 'demo', 'outside_hp_ssl.png');
let overlay, backend, tray;
let quitting = false;

function startBackend() {
  const py = process.env.DEMHIST_PYTHON || path.join(REPO, '.venv', 'Scripts', 'python.exe');
  backend = spawn(py, ['server.py'], { cwd: path.join(REPO, 'backend'), env: process.env });
  backend.stdout.on('data', d => process.stdout.write(`[backend] ${d}`));
  backend.stderr.on('data', d => process.stdout.write(`[backend] ${d}`));
  backend.on('exit', code => console.log(`[backend] exited ${code}`));
}

// all text and controls at 90%: the overlay code works in CSS px and reads devicePixelRatio, so the screenshot,
// selection and heatmap stay pixel-exact at any zoom
const UI_SCALE = 0.9;

const alive = w => w && !w.isDestroyed();

// true-colour capture through the Python backend (Pillow/GDI, ~0.15 s); null if the backend isn't up yet
async function backendCapture(rect) {
  try {
    const fd = new FormData();
    for (const k of ['x', 'y']) fd.append(k, String(rect[k]));
    fd.append('w', String(rect.width)); fd.append('h', String(rect.height));
    const r = await fetch('http://127.0.0.1:8765/capture', { method: 'POST', body: fd, signal: AbortSignal.timeout(1500) });
    const { path: file } = await r.json();
    return file && fs.existsSync(file) ? file : null;
  } catch { return null; }
}

async function captureScreen() {
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  const rect = screen.dipToScreenRect(null, display.bounds);
  // true-colour capture through the backend; desktopCapturer only as a last resort, because it
  // contrast-stretches the screen. (No PowerShell helper: antivirus stops hidden PowerShell processes.)
  // The overlay gets the raw PNG bytes (encoding a full-screen screenshot as a data URL took a noticeable moment).
  const file = (await backendCapture(rect)) || (await backendCapture(rect));  // one retry while the backend starts
  if (file) return { display, shot: fs.readFileSync(file) };  // raw PNG bytes; the overlay makes a blob URL
  const { width, height } = display.size;
  const sf = display.scaleFactor;
  const sources = await desktopCapturer.getSources({
    types: ['screen'], thumbnailSize: { width: Math.round(width * sf), height: Math.round(height * sf) },
  });
  const src = sources.find(s => s.display_id === String(display.id)) || sources[0];
  return { display, shot: src.thumbnail.toPNG() };
}

let opening = false;
async function openOverlay() {
  if (alive(overlay)) return focusOverlay();
  if (opening) return;  // hotkey pressed twice while the screen is being captured
  opening = true;
  try {
    const { display, shot } = await captureScreen();
    // keep a local handle: callbacks must never touch a newer overlay or one that is already closed
    // Hidden until the page has drawn the screenshot (it says so with 'overlay-ready'), so it appears as the screen
    // itself with the splash card on top: no flash of an empty window.
    const win = new BrowserWindow({
      ...display.bounds, frame: false, alwaysOnTop: true, skipTaskbar: true, fullscreen: true, show: false,
      backgroundColor: '#1b1c1e', webPreferences: { preload: path.join(__dirname, 'preload.js'), zoomFactor: UI_SCALE },
    });
    overlay = win;
    const reveal = () => {
      if (!alive(win) || win.isVisible()) return;
      win.show(); win.moveTop(); win.focus();
      // a window opened from a global hotkey often doesn't get keyboard focus on Windows, so its own Esc handler
      // never fires: Esc is taken globally while the snip window is open
      globalShortcut.register('Escape', () => { if (alive(win)) win.close(); });
    };
    const fallback = setTimeout(reveal, 2000);
    win.__reveal = () => { clearTimeout(fallback); reveal(); };
    // the page's warnings and errors in this terminal, next to the backend's log
    win.webContents.on('console-message', (_e, level, message, line) => { if (level >= 1) console.log(`[overlay] ${message} (line ${line})`); });
    win.webContents.on('render-process-gone', (_e, d) => console.log(`[overlay] renderer gone: ${d.reason}`));
    // links (PathologyOutlines, DOIs) open in the default browser
    win.webContents.setWindowOpenHandler(({ url }) => { if (/^https:\/\//.test(url)) shell.openExternal(url); return { action: 'deny' }; });
    win.loadFile('overlay.html');
    win.webContents.once('did-finish-load', () => {
      if (alive(win)) win.webContents.send('init', { screenshot: shot, researcher: '' });
    });
    win.on('closed', () => { globalShortcut.unregister('Escape'); if (overlay === win) overlay = null; });
  } finally { opening = false; }
}

// e.g. after Alt+Tab away from an open snip: bring it back instead of ignoring the hotkey
function focusOverlay() {
  if (overlay.isMinimized()) overlay.restore();
  overlay.show();
  overlay.focus();
}

ipcMain.on('overlay-ready', e => { const w = BrowserWindow.fromWebContents(e.sender); if (w && w.__reveal) w.__reveal(); });
ipcMain.on('overlay-close', () => { if (alive(overlay)) overlay.close(); });

app.whenReady().then(() => {
  startBackend();
  globalShortcut.register(HOTKEY, () => openOverlay());
  tray = new Tray(nativeImage.createEmpty());
  tray.setToolTip(`gutfeeling — ${HOTKEY.replace('CommandOrControl', 'Ctrl')}`);
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open (Ctrl+Shift+Space)', click: () => openOverlay() },
    { label: 'Quit', click: () => app.quit() },
  ]));
  // start silently: nothing shows until the hotkey (set DEMHIST_DEMO=1 to also open the demo sheet in the photo viewer)
  if (process.env.DEMHIST_DEMO && fs.existsSync(DEMO_IMAGE)) shell.openPath(DEMO_IMAGE);
});

app.on('window-all-closed', e => e.preventDefault()); // stay in the tray
app.on('before-quit', () => { quitting = true; });
app.on('will-quit', () => { globalShortcut.unregisterAll(); if (backend) backend.kill(); });
