// Compatibility wrapper. The verified deck builder is Python-based so it can
// use the same pinned environment as the experiments.
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const localPython = path.join(root, '.venv', 'Scripts', 'python.exe');
const python = fs.existsSync(localPython) ? localPython : (process.env.PYTHON || 'python');
const result = spawnSync(python, [path.join(__dirname, 'build_deck.py')], {
  cwd: root,
  stdio: 'inherit',
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
