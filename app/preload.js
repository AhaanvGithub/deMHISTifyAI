const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('demhist', {
  close: () => ipcRenderer.send('overlay-close'),
  onInit: cb => ipcRenderer.on('init', (_e, d) => cb(d)),
  ready: () => ipcRenderer.send('overlay-ready'),  // screenshot drawn: show the window
  backend: 'http://127.0.0.1:8765',
});
