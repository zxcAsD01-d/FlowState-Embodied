const registry = new Map();

const SCENE_VERSION = '20261028d';

export async function getSceneFactory(id) {
  const loader = registry.get(id)
    || (() => import(`/scenes/${id}.js?v=${SCENE_VERSION}`).then(m => m.createScene));
  try {
    return await loader();
  } catch (err) {
    throw new Error(`scene module "${id}" is not registered: ${err.message}`);
  }
}

export function registerScene(id, factory) {
  registry.set(id, async () => factory);
}
