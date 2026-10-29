import * as THREE from 'three';
import { buildArena } from './arena.js';

const W = 320;
const H = 180;
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({
  canvas, antialias: false, alpha: false, preserveDrawingBuffer: true,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;

let scene = null;
let camera = null;
let arena = null;

function disposeScene() {
  if (!scene) return;
  scene.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach(m => m && m.dispose && m.dispose());
    }
  });
  scene = null;
}

window.setScene = (spec) => {
  disposeScene();
  scene = new THREE.Scene();
  arena = buildArena(scene, spec, { background: '#0b111b' });
  camera = new THREE.PerspectiveCamera(72, W / H, 0.05, 80);
  camera.layers.set(0);
  return { ok: true, half: arena.half, boxes: arena.boxes.length };
};

window.renderFrames = (frames) => {
  if (!scene || !camera) return [];
  const out = [];
  for (const f of frames) {
    const eye = f.eye;
    const yaw = Number(f.yaw || 0);
    const pitch = Number(f.pitch || 0);
    camera.position.set(eye[0], eye[1], eye[2]);
    const fwd = new THREE.Vector3(
      Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch),
      Math.cos(yaw) * Math.cos(pitch));
    camera.up.set(0, 1, 0);
    camera.lookAt(camera.position.clone().add(fwd));
    renderer.render(scene, camera);
    out.push(canvas.toDataURL('image/jpeg', 0.88));
  }
  return out;
};

window.__ready = true;
