const input = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const preview = document.getElementById('preview');
const uploadContent = document.getElementById('uploadContent');
const detectBtn = document.getElementById('detectBtn');
const statusEl = document.getElementById('status');
const result = document.getElementById('result');
const plantName = document.getElementById('plantName');
const diseaseName = document.getElementById('diseaseName');
const confidence = document.getElementById('confidence');
const recommendation = document.getElementById('recommendation');
let selectedImage = null;
let model = null;
let labels = null;

function setImage(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedImage = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  uploadContent.hidden = true;
  detectBtn.disabled = false;
  result.hidden = true;
  statusEl.textContent = 'Image ready. Click Analyze Leaf.';
}

input.addEventListener('change', e => setImage(e.target.files[0]));
['dragenter','dragover'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.add('drag'); }));
['dragleave','drop'].forEach(event => dropZone.addEventListener(event, e => { e.preventDefault(); dropZone.classList.remove('drag'); }));
dropZone.addEventListener('drop', e => setImage(e.dataTransfer.files[0]));

async function loadAssets() {
  if (!model) model = await tf.loadLayersModel('model/model.json');
  if (!labels) labels = await fetch('class_labels.json').then(r => r.json());
}

function pretty(value) {
  return value.replace(/___/g, ' — ').replace(/_/g, ' ').replace(/\(including sour\)/g, '(including sour)').replace(/, bell/g, ', bell');
}

detectBtn.addEventListener('click', async () => {
  if (!selectedImage) return;
  detectBtn.disabled = true;
  statusEl.textContent = 'Loading AI model and analyzing…';
  try {
    await loadAssets();
    const img = new Image();
    img.src = URL.createObjectURL(selectedImage);
    await img.decode();
    const inputTensor = tf.tidy(() => tf.browser.fromPixels(img).resizeBilinear([224, 224]).toFloat().div(255).expandDims(0));
    const output = model.predict(inputTensor);
    const values = await output.data();
    const index = values.indexOf(Math.max(...values));
    const score = values[index] * 100;
    const raw = labels[index];
    const parts = raw.split('___');
    const plant = parts[0] || 'Unknown plant';
    const disease = parts[1] || raw;
    plantName.textContent = plant.replace(/_/g, ' ');
    diseaseName.textContent = disease.replace(/_/g, ' ');
    confidence.textContent = `${score.toFixed(1)}% confidence`;
    recommendation.innerHTML = raw.toLowerCase().includes('healthy')
      ? '✅ <strong>Healthy:</strong> The model classified this leaf as healthy. Continue regular monitoring and good crop care.'
      : '⚠️ <strong>Possible disease detected:</strong> Use this result as a screening aid and confirm the diagnosis with a local agricultural expert before treatment.';
    result.hidden = false;
    statusEl.textContent = 'Analysis complete.';
    tf.dispose([inputTensor, output]);
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'The AI model is not connected yet. Add the converted TensorFlow.js model in the model/ folder.';
  } finally {
    detectBtn.disabled = false;
  }
});

document.getElementById('resetBtn').addEventListener('click', () => {
  selectedImage = null;
  input.value = '';
  preview.hidden = true;
  uploadContent.hidden = false;
  detectBtn.disabled = true;
  result.hidden = true;
  statusEl.textContent = '';
});
