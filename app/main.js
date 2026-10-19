// deMHISTifyAI desktop shell.
// Ctrl+Shift+Space -> Spotlight-style bar -> type your call -> Enter -> snip the screen -> results.
const { app, BrowserWindow, globalShortcut, desktopCapturer, screen, ipcMain, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const capture = require('./capture');

const HOTKEY = 'CommandOrControl+Shift+Space';
const REPO = path.resolve(__dirname, '..');
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

function showLauncher() {
  if (!alive(launcher)) createLauncher();
  launcher.show();
  launcher.focus();
  launcher.webContents.send('focus');
}

async function captureScreen() {
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  // true-colour capture first (see capture.js); desktopCapturer only as a fallback
  const file = await capture.grab(screen.dipToScreenRect(null, display.bounds));
  if (file) return { display, dataUrl: nativeImage.createFromPath(file).toDataURL() };
  const { width, height } = display.size;
  const sf = display.scaleFactor;
  const sources = await desktopCapturer.getSources({
    types: ['screen'], thumbnailSize: { width: Math.round(width * sf), height: Math.round(height * sf) },
  });
  const src = sources.find(s => s.display_id === String(display.id)) || sources[0];
  return { display, dataUrl: src.thumbnail.toDataURL() };
}

async function openOverlay() {
  if (alive(overlay)) return focusOverlay();
  const { display, dataUrl } = await captureScreen();
  // keep a local handle: callbacks must never touch a newer overlay or one that is already closed
  const win = new BrowserWindow({
    ...display.bounds, frame: false, alwaysOnTop: true, skipTaskbar: true, fullscreen: true,
    backgroundColor: '#170a25', webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  overlay = win;
  win.loadFile('overlay.html');
  win.webContents.once('did-finish-load', () => {
    if (alive(win)) win.webContents.send('init', { screenshot: dataUrl, researcher: researcherText });
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
  capture.start();  // warm up the capture helper so the first snip is fast
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
  launcher.webContents.once('did-finish-load', showLauncher);
});

app.on('window-all-closed', e => e.preventDefault()); // stay in the tray
app.on('before-quit', () => { quitting = true; });
app.on('will-quit', () => { globalShortcut.unregisterAll(); capture.stop(); if (backend) backend.kill(); });
