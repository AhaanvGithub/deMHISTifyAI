const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('demhist', {
  submit: text => ipcRenderer.send('launcher-submit', text),
  close: () => ipcRenderer.send('overlay-close'),
  onInit: cb => ipcRenderer.on('init', (_e, d) => cb(d)),
  onFocus: cb => ipcRenderer.on('focus', () => cb()),
  backend: 'http://127.0.0.1:8765',
});
