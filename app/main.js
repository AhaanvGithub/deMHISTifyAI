// deMHISTifyAI desktop shell.
// Ctrl+Shift+Space -> Spotlight-style bar -> type your call -> Enter -> snip the screen -> results.
const { app, BrowserWindow, globalShortcut, desktopCapturer, screen, ipcMain, Tray, Menu, nativeImage, shell } = require('electron');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const HOTKEY = 'CommandOrControl+Shift+Space';
const REPO = path.resolve(__dirname, '..');
// two 448 um fields (MHIST's field size), built by research/make_demo_image.py; set DEMHIST_NO_DEMO=1 to skip
const DEMO_IMAGE = path.join(REPO, 'data', 'demo', 'demo_hp_ssl.png');
let launcher, overlay, backend, tray;
let researcherText = '';
let quitting = false;

function startBackend() {
  const py = process.env.DEMHIST_PYTHON || path.join(REPO, '.venv', 'Scripts', 'python.exe');
  backend = spawn(py, ['server.py'], { cwd: path.join(REPO, 'backend'), env: process.env });
  backend.stdout.on('data', d => process.stdout.write(`[backend] ${d}`));
  backend.stderr.on('data', d => process.stdout.write(`[backend] ${d}`));
  backend.on('exit', code => console.log(`[backend] exited ${code}`));
}

function createLauncher() {
  const { width } = screen.getPrimaryDisplay().workAreaSize;
  launcher = new BrowserWindow({
    width: 860, height: 124, x: Math.round((width - 860) / 2), y: 160, // pill + 28px margin for its glow
    // no OS frame/shadow behind the frameless transparent window (it shows up as a square outline)
    hasShadow: false, thickFrame: false, backgroundColor: '#00000000',
    frame: false, transparent: true, resizable: false, alwaysOnTop: true, show: false, skipTaskbar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  launcher.loadFile('launcher.html');
  launcher.on('blur', () => launcher.hide());
  // Esc in the launcher calls window.close(): hide instead, or the hotkey later hits a destroyed window
  launcher.on('close', e => { if (!quitting) { e.preventDefault(); launcher.hide(); } });
}

const alive = w => w && !w.isDestroyed();

// The first show of a transparent window is slow (surfaces and GPU layers are created on demand), which made the
// first hotkey press stutter. Rehearse once at start-up: fully transparent, click-through, never focused.
let warmTimer = null;
function warmLauncher() {
  if (!alive(launcher)) return;
  launcher.setOpacity(0);
  launcher.setIgnoreMouseEvents(true);
  launcher.showInactive();
  launcher.webContents.send('warm');  // plays the entrance animation once, then parks the pill
  warmTimer = setTimeout(endWarm, 600);
}
function endWarm(keepShown = false) {
  clearTimeout(warmTimer); warmTimer = null;
  if (!alive(launcher)) return;
  if (!keepShown) launcher.hide();
  launcher.setIgnoreMouseEvents(false);
  launcher.setOpacity(1);
}

function showLauncher() {
  if (!alive(launcher)) createLauncher();
  if (warmTimer) endWarm(true);  // hotkey pressed during the start-up rehearsal
  launcher.show();
  launcher.focus();
  launcher.webContents.send('focus');
}

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

async function openOverlay() {
  if (alive(overlay)) return focusOverlay();
  const { display, shot } = await captureScreen();
  // keep a local handle: callbacks must never touch a newer overlay or one that is already closed
  const win = new BrowserWindow({
    ...display.bounds, frame: false, alwaysOnTop: true, skipTaskbar: true, fullscreen: true,
    backgroundColor: '#170a25', webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  overlay = win;
  win.loadFile('overlay.html');
  win.webContents.once('did-finish-load', () => {
    if (alive(win)) win.webContents.send('init', { screenshot: shot, researcher: researcherText });
  });
  win.on('closed', () => { if (overlay === win) overlay = null; });
}

// e.g. after Alt+Tab away from an open snip: bring it back instead of ignoring the hotkey
function focusOverlay() {
  if (overlay.isMinimized()) overlay.restore();
  overlay.show();
  overlay.focus();
}

ipcMain.on('launcher-submit', async (_e, text) => {
  researcherText = text;
  if (alive(launcher)) launcher.hide();
  setTimeout(openOverlay, 180); // let the launcher fade before the screenshot
});
ipcMain.on('overlay-close', () => { if (alive(overlay)) overlay.close(); });

app.whenReady().then(() => {
  startBackend();
  createLauncher();
  globalShortcut.register(HOTKEY, () => {
    if (alive(overlay)) return focusOverlay();
    showLauncher();
  });
  tray = new Tray(nativeImage.createEmpty());
  tray.setToolTip(`deMHISTifyAI — ${HOTKEY.replace('CommandOrControl', 'Ctrl')}`);
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open (Ctrl+Shift+Space)', click: () => showLauncher() },
    { label: 'Quit', click: () => app.quit() },
  ]));
  launcher.webContents.once('did-finish-load', warmLauncher);
  // start silently: nothing shows until the hotkey. Open the demo slide image in the default photo viewer.
  if (!process.env.DEMHIST_NO_DEMO && fs.existsSync(DEMO_IMAGE)) shell.openPath(DEMO_IMAGE);
});

app.on('window-all-closed', e => e.preventDefault()); // stay in the tray
app.on('before-quit', () => { quitting = true; });
app.on('will-quit', () => { globalShortcut.unregisterAll(); if (backend) backend.kill(); });
