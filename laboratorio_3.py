import numpy as np


# =============================================================================
# Tarefa 1 - Mascara Causal (Look-Ahead Mask)
# =============================================================================

def create_causal_mask(seq_len):
    """Cria uma mascara causal de tamanho [seq_len, seq_len].

    A triangular inferior (incluindo diagonal) fica com zeros,
    enquanto a triangular superior recebe -inf para bloquear
    posicoes futuras antes do Softmax.
    """
    mask = np.full((seq_len, seq_len), -np.inf)
    mask = np.tril(np.zeros((seq_len, seq_len))) + np.triu(mask, k=1)
    return mask


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def demo_causal_mask():
    seq_len = 5
    d_k = 8

    np.random.seed(0)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    scores = Q @ K.T / np.sqrt(d_k)
    M = create_causal_mask(seq_len)
    masked_scores = scores + M
    attn_weights = softmax(masked_scores)

    print("=== Tarefa 1: Atencao com Mascara Causal ===")
    print("Pesos de atencao (posicoes futuras devem ser 0.0):")
    print(np.round(attn_weights, 4))
    assert np.allclose(np.triu(attn_weights, k=1), 0.0), "Falha: posicoes futuras nao sao zero!"
    print("OK - Probabilidades de posicoes futuras sao estritamente 0.0\n")


# =============================================================================
# Tarefa 2 - Cross-Attention (Ponte Encoder-Decoder)
# =============================================================================

def cross_attention(encoder_out, decoder_state):
    """Calcula o Scaled Dot-Product Attention cruzando encoder e decoder.

    Query vem do decoder_state; Keys e Values vem do encoder_out.
    Sem mascara causal aqui, pois o decoder pode ver o encoder inteiro.
    """
    batch, seq_enc, d_model = encoder_out.shape
    _, seq_dec, _ = decoder_state.shape

    np.random.seed(42)
    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01

    Q = decoder_state @ W_q   # [1, seq_dec, d_model]
    K = encoder_out @ W_k     # [1, seq_enc, d_model]
    V = encoder_out @ W_v     # [1, seq_enc, d_model]

    d_k = d_model
    # scores: [1, seq_dec, seq_enc]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V)  # [1, seq_dec, d_model]
    return output, attn_weights


def demo_cross_attention():
    batch_size = 1
    seq_len_enc = 10
    seq_len_dec = 4
    d_model = 512

    np.random.seed(7)
    encoder_output = np.random.randn(batch_size, seq_len_enc, d_model)
    decoder_state = np.random.randn(batch_size, seq_len_dec, d_model)

    output, attn_weights = cross_attention(encoder_output, decoder_state)

    print("=== Tarefa 2: Cross-Attention ===")
    print(f"encoder_output shape : {encoder_output.shape}")
    print(f"decoder_state shape  : {decoder_state.shape}")
    print(f"output shape         : {output.shape}")
    print(f"attn_weights shape   : {attn_weights.shape}")
    print(f"Soma das atencoes (deve ser ~1 por linha): {attn_weights[0].sum(axis=-1).round(4)}\n")


# =============================================================================
# Tarefa 3 - Loop de Inferencia Auto-Regressivo
# =============================================================================

VOCAB_SIZE = 10_000
EOS_IDX = 9999
EOS_TOKEN = "<EOS>"

# Vocabulario ficticio
vocab = [f"token_{i}" for i in range(VOCAB_SIZE)]
vocab[EOS_IDX] = EOS_TOKEN
vocab[0] = "<START>"


def generate_next_token(current_sequence, encoder_out):
    """Mock do passo de decodificacao.

    Simula a saida do Decoder: projeta o estado atual para o
    tamanho do vocabulario e retorna uma distribuicao de probabilidades.
    """
    np.random.seed(len(current_sequence))  # seed variavel por comprimento
    logits = np.random.randn(VOCAB_SIZE)

    # Pequeno truque: conforme a sequencia fica longa, aumentamos a
    # probabilidade do EOS para o loop terminar de forma natural.
    if len(current_sequence) >= 6:
        logits[EOS_IDX] += 5.0

    probs = softmax(logits)
    return probs


def demo_autoregressive_loop():
    batch_size = 1
    seq_len_enc = 10
    d_model = 512

    np.random.seed(1)
    encoder_out = np.random.randn(batch_size, seq_len_enc, d_model)

    sequence = ["<START>"]
    max_steps = 20

    print("=== Tarefa 3: Loop Auto-Regressivo ===")
    print(f"Iniciando geracao a partir de: {sequence}")

    step = 0
    while step < max_steps:
        probs = generate_next_token(sequence, encoder_out)
        next_idx = int(np.argmax(probs))
        next_token = vocab[next_idx]

        sequence.append(next_token)
        print(f"  Passo {step + 1}: token gerado = '{next_token}' (idx={next_idx}, prob={probs[next_idx]:.4f})")

        if next_token == EOS_TOKEN:
            break
        step += 1

    print(f"\nFrase final gerada: {' '.join(sequence)}\n")


# =============================================================================
# Execucao principal
# =============================================================================

if __name__ == "__main__":
    demo_causal_mask()
    demo_cross_attention()
    demo_autoregressive_loop()
