import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The most atomic way to train and inference a GPT in pure Java.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * <p>
 * Translated from @karpathy's Python implementation
 */
public class MinimalGPT {

    private static final Random random = new Random(42); // Let there be order among chaos

    public static void main(String[] args) throws Exception {

        // Let there be an input dataset `docs`: List<String> of documents (e.g. a dataset of names)
        if (!Files.exists(Paths.get("input.txt"))) {
            String namesUrl = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
            try (InputStream in = URI.create(namesUrl).toURL().openStream()) {
                Files.copy(in, Paths.get("input.txt"));
            }
        }

        List<String> docs = Files.readAllLines(Paths.get("input.txt")).stream()
                .map(String::strip)
                .filter(l -> !l.isEmpty())
                .collect(Collectors.toList());
        Collections.shuffle(docs, random);
        System.out.println("num docs: " + docs.size());

        // Let there be a Tokenizer to translate strings to discrete symbols and back
        List<Character> uchars = docs.stream()
                .flatMapToInt(String::chars)
                .distinct()
                .sorted()
                .mapToObj(c -> (char) c)
                .toList();
        int BOS = uchars.size(); // token id for the special Beginning of Sequence (BOS) token
        int vocabSize = uchars.size() + 1; // total number of unique tokens, +1 is for BOS
        System.out.println("vocab size: " + vocabSize);

        // Initialize the parameters, to store the knowledge of the model
        int nEmbd = 16;      // embedding dimension
        int nHead = 4;       // number of attention heads
        int nLayer = 1;      // number of layers
        int blockSize = 16;  // maximum sequence length
        int headDim = nEmbd / nHead; // dimension of each head

        Map<String, Value[][]> stateDict = new HashMap<>();
        stateDict.put("wte", matrix(vocabSize, nEmbd, 0.08));
        stateDict.put("wpe", matrix(blockSize, nEmbd, 0.08));
        stateDict.put("lm_head", matrix(vocabSize, nEmbd, 0.08));

        for (int i = 0; i < nLayer; i++) {
            stateDict.put("layer" + i + ".attn_wq", matrix(nEmbd, nEmbd, 0.08));
            stateDict.put("layer" + i + ".attn_wk", matrix(nEmbd, nEmbd, 0.08));
            stateDict.put("layer" + i + ".attn_wv", matrix(nEmbd, nEmbd, 0.08));
            stateDict.put("layer" + i + ".attn_wo", matrix(nEmbd, nEmbd, 0.08));
            stateDict.put("layer" + i + ".mlp_fc1", matrix(4 * nEmbd, nEmbd, 0.08));
            stateDict.put("layer" + i + ".mlp_fc2", matrix(nEmbd, 4 * nEmbd, 0.08));
        }

        // flatten params into a single List<Value>
        List<Value> params = stateDict.values().stream()
                .flatMap(Arrays::stream)
                .flatMap(Arrays::stream)
                .toList();
        System.out.println("num params: " + params.size());

        // Let there be Adam, the blessed optimizer and its buffers
        double learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, epsAdam = 1e-8;
        double[] m = new double[params.size()]; // first moment buffer
        double[] v = new double[params.size()]; // second moment buffer

        // Repeat in sequence
        int numSteps = 1000; // number of training steps
        for (int step = 0; step < numSteps; step++) {

            // Take single document, tokenize it, surround it with BOS special token on both sides
            String doc = docs.get(step % docs.size());
            List<Integer> tokens = new ArrayList<>();
            tokens.add(BOS);
            for (char ch : doc.toCharArray()) {
                tokens.add(uchars.indexOf(ch));
            }
            tokens.add(BOS);
            int n = Math.min(blockSize, tokens.size() - 1);

            // Forward the token sequence through the model, building up the computation graph all the way to the loss
            List<List<Value[]>> keys = new ArrayList<>();
            List<List<Value[]>> values = new ArrayList<>();
            for (int i = 0; i < nLayer; i++) {
                keys.add(new ArrayList<>());
                values.add(new ArrayList<>());
            }

            List<Value> losses = new ArrayList<>();
            for (int posId = 0; posId < n; posId++) {
                int tokenId = tokens.get(posId);
                int targetId = tokens.get(posId + 1);
                Value[] logits = gpt(tokenId, posId, keys, values, stateDict, nLayer, nEmbd, nHead, headDim);
                Value[] probs = softmax(logits);
                Value lossT = probs[targetId].log().neg();
                losses.add(lossT);
            }
            Value loss = losses.stream()
                    .reduce(Value::add)
                    .map(sum -> sum.div(n))
                    .get(); // final average loss over the document sequence. May yours be low.

            // Backward the loss, calculating the gradients with respect to all model parameters
            loss.backward();

            // Adam optimizer update: update the model parameters based on the corresponding gradients
            double lrT = learningRate * (1.0 - (double) step / numSteps); // linear learning rate decay
            for (int i = 0; i < params.size(); i++) {
                Value p = params.get(i);
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
                v[i] = beta2 * v[i] + (1 - beta2) * Math.pow(p.grad, 2);
                double mHat = m[i] / (1 - Math.pow(beta1, step + 1));
                double vHat = v[i] / (1 - Math.pow(beta2, step + 1));
                p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
                p.grad = 0;
            }

            System.out.printf("step %4d / %4d | loss %.4f%n", step + 1, numSteps, loss.data);
        }

        // Inference: may the model babble back to us
        double temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
        System.out.println("\n--- inference (new, hallucinated names) ---");
        for (int sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
            List<List<Value[]>> keys = new ArrayList<>();
            List<List<Value[]>> values = new ArrayList<>();
            for (int i = 0; i < nLayer; i++) {
                keys.add(new ArrayList<>());
                values.add(new ArrayList<>());
            }

            int tokenId = BOS;
            StringBuilder sample = new StringBuilder();
            for (int posId = 0; posId < blockSize; posId++) {
                Value[] logits = gpt(tokenId, posId, keys, values, stateDict, nLayer, nEmbd, nHead, headDim);
                Value[] scaledLogits = Arrays.stream(logits)
                        .map(l -> l.div(temperature))
                        .toArray(Value[]::new);
                Value[] probs = softmax(scaledLogits);
                double[] weights = Arrays.stream(probs).mapToDouble(p -> p.data).toArray();
                tokenId = weightedChoice(IntStream.range(0, vocabSize).toArray(), weights);
                if (tokenId == BOS) {
                    break;
                }
                sample.append(uchars.get(tokenId));
            }
            System.out.printf("sample %2d: %s%n", sampleIdx + 1, sample);
        }
    }

    // Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next
    // Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
    private static Value[] linear(Value[] x, Value[][] w) {
        Value[] result = new Value[w.length];
        for (int i = 0; i < w.length; i++) {
            Value sum = new Value(0);
            for (int j = 0; j < x.length; j++) {
                sum = sum.add(w[i][j].mul(x[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    private static Value[] softmax(Value[] logits) {
        double maxVal = Arrays.stream(logits).mapToDouble(v -> v.data).max().getAsDouble();
        Value[] exps = Arrays.stream(logits)
                .map(val -> val.sub(maxVal).exp())
                .toArray(Value[]::new);
        Value total = Arrays.stream(exps).reduce(Value::add).get();
        return Arrays.stream(exps).map(e -> e.div(total)).toArray(Value[]::new);
    }

    private static Value[] rmsnorm(Value[] x) {
        Value ms = Arrays.stream(x)
                .map(xi -> xi.mul(xi))
                .reduce(Value::add)
                .map(sum -> sum.div(x.length))
                .get();
        Value scale = ms.add(1e-5).pow(-0.5);
        return Arrays.stream(x).map(xi -> xi.mul(scale)).toArray(Value[]::new);
    }

    private static Value[] gpt(int tokenId, int posId, List<List<Value[]>> keys, List<List<Value[]>> values,
                               Map<String, Value[][]> stateDict, int nLayer, int nEmbd, int nHead, int headDim) {
        Value[] tokEmb = stateDict.get("wte")[tokenId]; // token embedding
        Value[] posEmb = stateDict.get("wpe")[posId]; // position embedding
        Value[] x = new Value[nEmbd];
        for (int i = 0; i < nEmbd; i++) {
            x[i] = tokEmb[i].add(posEmb[i]); // joint token and position embedding
        }
        x = rmsnorm(x);

        for (int li = 0; li < nLayer; li++) {
            // 1) Multi-head attention block
            Value[] xResidual = x;
            x = rmsnorm(x);
            Value[] q = linear(x, stateDict.get("layer" + li + ".attn_wq"));
            Value[] k = linear(x, stateDict.get("layer" + li + ".attn_wk"));
            Value[] vVec = linear(x, stateDict.get("layer" + li + ".attn_wv"));
            keys.get(li).add(k);
            values.get(li).add(vVec);

            List<Value> xAttn = new ArrayList<>();
            for (int h = 0; h < nHead; h++) {
                int hs = h * headDim;
                Value[] qH = Arrays.copyOfRange(q, hs, hs + headDim);
                List<Value[]> kH = keys.get(li).stream()
                        .map(ki -> Arrays.copyOfRange(ki, hs, hs + headDim))
                        .toList();
                List<Value[]> vH = values.get(li).stream()
                        .map(vi -> Arrays.copyOfRange(vi, hs, hs + headDim))
                        .toList();

                Value[] attnLogits = new Value[kH.size()];
                for (int t = 0; t < kH.size(); t++) {
                    Value sum = new Value(0);
                    for (int j = 0; j < headDim; j++) {
                        sum = sum.add(qH[j].mul(kH.get(t)[j]));
                    }
                    attnLogits[t] = sum.div(Math.sqrt(headDim));
                }
                Value[] attnWeights = softmax(attnLogits);

                for (int j = 0; j < headDim; j++) {
                    Value sum = new Value(0);
                    for (int t = 0; t < vH.size(); t++) {
                        sum = sum.add(attnWeights[t].mul(vH.get(t)[j]));
                    }
                    xAttn.add(sum);
                }
            }
            x = linear(xAttn.toArray(new Value[0]), stateDict.get("layer" + li + ".attn_wo"));
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].add(xResidual[i]);
            }

            // 2) MLP block
            xResidual = x;
            x = rmsnorm(x);
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc1"));
            x = Arrays.stream(x).map(Value::relu).toArray(Value[]::new);
            x = linear(x, stateDict.get("layer" + li + ".mlp_fc2"));
            for (int i = 0; i < x.length; i++) {
                x[i] = x[i].add(xResidual[i]);
            }
        }

        return linear(x, stateDict.get("lm_head"));
    }

    // Helper methods
    private static Value[][] matrix(int nout, int nin, double std) {
        Value[][] mat = new Value[nout][nin];
        for (int i = 0; i < nout; i++) {
            for (int j = 0; j < nin; j++) {
                mat[i][j] = new Value(random.nextGaussian() * std);
            }
        }
        return mat;
    }

    private static int weightedChoice(int[] choices, double[] weights) {
        double total = Arrays.stream(weights).sum();
        double r = random.nextDouble() * total;
        double cumulative = 0;
        for (int i = 0; i < choices.length; i++) {
            cumulative += weights[i];
            if (r <= cumulative) {
                return choices[i];
            }
        }
        return choices[choices.length - 1];
    }

    // Let there be Autograd, to recursively apply the chain rule through a computation graph
    static class Value {
        double data;              // scalar value of this node calculated during forward pass
        double grad;              // derivative of the loss w.r.t. this node, calculated in backward pass
        List<Value> children;     // children of this node in the computation graph
        List<Double> localGrads;  // local derivative of this node w.r.t. its children

        public Value(double data) {
            this(data, new ArrayList<>(), new ArrayList<>());
        }

        public Value(double data, List<Value> children, List<Double> localGrads) {
            this.data = data;
            this.grad = 0;
            this.children = children;
            this.localGrads = localGrads;
        }

        public Value add(Value other) {
            return new Value(
                    this.data + other.data,
                    Arrays.asList(this, other),
                    Arrays.asList(1.0, 1.0)
            );
        }

        public Value add(double other) {
            return add(new Value(other));
        }

        public Value mul(Value other) {
            return new Value(
                    this.data * other.data,
                    Arrays.asList(this, other),
                    Arrays.asList(other.data, this.data)
            );
        }

        public Value mul(double other) {
            return mul(new Value(other));
        }

        public Value pow(double other) {
            return new Value(
                    Math.pow(this.data, other),
                    Collections.singletonList(this),
                    Collections.singletonList(other * Math.pow(this.data, other - 1))
            );
        }

        public Value log() {
            return new Value(
                    Math.log(this.data),
                    Collections.singletonList(this),
                    Collections.singletonList(1.0 / this.data)
            );
        }

        public Value exp() {
            double expVal = Math.exp(this.data);
            return new Value(
                    expVal,
                    Collections.singletonList(this),
                    Collections.singletonList(expVal)
            );
        }

        public Value relu() {
            return new Value(
                    Math.max(0, this.data),
                    Collections.singletonList(this),
                    Collections.singletonList(this.data > 0 ? 1.0 : 0.0)
            );
        }

        public Value neg() {
            return mul(-1);
        }

        public Value sub(Value other) {
            return add(other.neg());
        }

        public Value sub(double other) {
            return add(-other);
        }

        public Value div(Value other) {
            return mul(other.pow(-1));
        }

        public Value div(double other) {
            return mul(Math.pow(other, -1));
        }

        public void backward() {
            List<Value> topo = new ArrayList<>();
            Set<Value> visited = new HashSet<>();
            buildTopo(this, visited, topo);

            this.grad = 1;
            for (int i = topo.size() - 1; i >= 0; i--) {
                Value v = topo.get(i);
                for (int j = 0; j < v.children.size(); j++) {
                    Value child = v.children.get(j);
                    double localGrad = v.localGrads.get(j);
                    child.grad += localGrad * v.grad;
                }
            }
        }

        private void buildTopo(Value v, Set<Value> visited, List<Value> topo) {
            if (!visited.contains(v)) {
                visited.add(v);
                for (Value child : v.children) {
                    buildTopo(child, visited, topo);
                }
                topo.add(v);
            }
        }
    }
}
