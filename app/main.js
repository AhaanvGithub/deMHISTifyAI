// deMHISTifyAI desktop shell.
// Ctrl+Shift+Space -> Spotlight-style bar -> type your call -> Enter -> snip the screen -> results.
const { app, BrowserWindow, globalShortcut, desktopCapturer, screen, ipcMain, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const HOTKEY = 'CommandOrControl+Shift+Space';
const REPO = path.resolve(__dirname, '..');
let launcher, overlay, backend, tray;
let researcherText = '';

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
    width: 680, height: 92, x: Math.round((width - 680) / 2), y: 180,
    frame: false, transparent: true, resizable: false, alwaysOnTop: true, show: false, skipTaskbar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  launcher.loadFile('launcher.html');
  launcher.on('blur', () => launcher.hide());
}

async function captureScreen() {
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  const { width, height } = display.size;
  const sf = display.scaleFactor;
  const sources = await desktopCapturer.getSources({
    types: ['screen'], thumbnailSize: { width: Math.round(width * sf), height: Math.round(height * sf) },
  });
  const src = sources.find(s => s.display_id === String(display.id)) || sources[0];
  return { display, dataUrl: src.thumbnail.toDataURL() };
}

async function openOverlay() {
  const { display, dataUrl } = await captureScreen();
  overlay = new BrowserWindow({
    ...display.bounds, frame: false, alwaysOnTop: true, skipTaskbar: true, fullscreen: true,
    backgroundColor: '#0b0d12', webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  overlay.loadFile('overlay.html');
  overlay.webContents.once('did-finish-load', () =>
    overlay.webContents.send('init', { screenshot: dataUrl, researcher: researcherText }));
  overlay.on('closed', () => (overlay = null));
}

ipcMain.on('launcher-submit', async (_e, text) => {
  researcherText = text;
  launcher.hide();
  setTimeout(openOverlay, 180); // let the launcher fade before the screenshot
});
ipcMain.on('overlay-close', () => overlay && overlay.close());

app.whenReady().then(() => {
  startBackend();
  createLauncher();
  globalShortcut.register(HOTKEY, () => {
    if (overlay) return;
    launcher.show();
    launcher.focus();
    launcher.webContents.send('focus');
  });
  tray = new Tray(nativeImage.createEmpty());
  tray.setToolTip(`deMHISTifyAI — ${HOTKEY.replace('CommandOrControl', 'Ctrl')}`);
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open (Ctrl+Shift+Space)', click: () => { launcher.show(); launcher.focus(); } },
    { label: 'Quit', click: () => app.quit() },
  ]));
  launcher.show();
});

app.on('window-all-closed', e => e.preventDefault()); // stay in the tray
app.on('will-quit', () => { globalShortcut.unregisterAll(); if (backend) backend.kill(); });
