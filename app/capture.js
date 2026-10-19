// Screen capture that keeps the screen's true colours.
// Electron's desktopCapturer returns a contrast-stretched image on this setup (measured: dark grey
// 21 -> 6, highlights 119 -> 153), which makes tissue look over-exposed. GDI's CopyFromScreen returns
// exactly what is displayed, so we keep one hidden PowerShell process warm and ask it for captures.
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const readline = require('readline');

const PS = `
Add-Type -AssemblyName System.Drawing
Add-Type 'using System.Runtime.InteropServices; public class Dpi { [DllImport("user32.dll")] public static extern bool SetProcessDPIAware(); }'
[Dpi]::SetProcessDPIAware() | Out-Null
while ($true) {
  $line = [Console]::In.ReadLine()
  if ($line -eq $null) { break }
  $a = $line.Split('|')
  try {
    $bmp = New-Object System.Drawing.Bitmap ([int]$a[3]), ([int]$a[4])
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen([int]$a[1], [int]$a[2], 0, 0, $bmp.Size)
    $bmp.Save($a[0], [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    [Console]::Out.WriteLine('ok')
  } catch { [Console]::Out.WriteLine('err ' + $_.Exception.Message) }
}`;

let proc = null, lines = null, waiting = [];

function start() {
  if (process.platform !== 'win32' || proc) return;
  proc = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(PS, 'utf16le').toString('base64')],
    { windowsHide: true });
  lines = readline.createInterface({ input: proc.stdout });
  lines.on('line', l => { const w = waiting.shift(); if (w) w(l.trim()); });
  proc.on('exit', () => { proc = null; waiting.splice(0).forEach(w => w('err exited')); });
}

// rect in physical screen pixels -> PNG path, or null if GDI capture isn't available
function grab(rect, timeoutMs = 4000) {
  start();
  if (!proc) return Promise.resolve(null);
  const file = path.join(os.tmpdir(), `demhist-snip-${process.pid}.png`);
  return new Promise(res => {
    const t = setTimeout(() => { waiting = waiting.filter(w => w !== done); res(null); }, timeoutMs);
    const done = reply => { clearTimeout(t); res(reply === 'ok' && fs.existsSync(file) ? file : null); };
    waiting.push(done);
    proc.stdin.write(`${file}|${rect.x}|${rect.y}|${rect.width}|${rect.height}\n`);
  });
}

function stop() { if (proc) { proc.stdin.end(); proc.kill(); proc = null; } }

module.exports = { start, grab, stop };
