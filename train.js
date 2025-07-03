const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { encode } = require('gpt-tokenizer'); // ADD THIS

// Panggil sebelum training
let bestValLoss = Number.POSITIVE_INFINITY;
let bestWeights = null;
let patience = 7;
let wait = 0;

const onEpochEnd = async (epoch, logs) => {
  if (logs.val_loss < bestValLoss) {
    bestValLoss = logs.val_loss;
    bestWeights = model.getWeights().map(w => w.clone());
    wait = 0;
  } else {
    wait++;
    if (wait >= patience) {
      console.log(`⛔️ Early stopping on epoch ${epoch + 1}`);
      // Restore best weights
      model.setWeights(bestWeights);
      throw 'early_stop'; // stop fit
    }
  }
};


// Dataset JSON format: [{ text: "...", label: "..." }]
const rawData = JSON.parse(fs.readFileSync('dataset.json', 'utf8'));
const texts = rawData.map(d => d.text);
const labels = rawData.map(d => d.label);

const labelSet = Array.from(new Set(labels));
const labelIndex = Object.fromEntries(labelSet.map((l, i) => [l, i]));
const numLabels = labelSet.length;

// Tokenizer config
const maxLen = 50;

// Gunakan tokenizer GPT, bukan split whitespace
function tokenize(text) {
  // encode() hasilkan array token id integer, sesuai tokenizer OpenAI
  const tokens = encode(text); // sudah case-insensitive
  if (tokens.length < maxLen)
    return tokens.concat(Array(maxLen - tokens.length).fill(0));
  return tokens.slice(0, maxLen);
}

// Cari vocab size dari tokenizer (id max + 1)
let maxTokenId = 0;
for (const t of texts) {
  for (const id of encode(t)) {
    if (id > maxTokenId) maxTokenId = id;
  }
}
const vocabSize = maxTokenId + 1;

// Tensorify
const xTrain = tf.tensor2d(texts.map(tokenize));
const yTrain = tf.tensor1d(labels.map(l => labelIndex[l]), 'float32');

// Build model
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: vocabSize, outputDim: 64, inputLength: maxLen }));
model.add(tf.layers.globalAveragePooling1d());
model.add(tf.layers.dense({ units: 126, activation: 'relu' }));
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: numLabels, activation: 'softmax' }));

model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy'],
});

(async () => {
  console.log("🚀 Training...");
  try {
    await model.fit(xTrain, yTrain, {
      epochs: 100,
      batchSize: 4,
      validationSplit: 0.2,
      shuffle: true,
      callbacks: { onEpochEnd },
    });
  } catch (e) {
    if (e !== 'early_stop') throw e; // lempar ulang kalau error lain
    // else: aman, early stopping
  }
  await model.save('file://./model');
  fs.writeFileSync('label_index.json', JSON.stringify(labelIndex, null, 2));
  console.log("✅ Training complete. Model saved.");
})();

