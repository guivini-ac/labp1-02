import numpy as np
import pandas as pd

np.random.seed(42)

# ════════════════════════════════════════════════
# PASSO 1: PREPARAÇÃO DOS DADOS
# ════════════════════════════════════════════════

# Vocabulário simulado via pandas
vocabulario = {
    "o": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartao": 3,
    "meu": 4,
    "ontem": 5,
}

df_vocab = pd.DataFrame(list(vocabulario.items()), columns=["palavra", "id"])
print("📋 Vocabulário carregado:")
print(df_vocab.to_string(index=False))

# Frase de entrada
frase = ["o", "banco", "bloqueou", "meu", "cartao"]
ids_frase = [vocabulario[palavra] for palavra in frase]
print(f"\n📝 Frase de entrada : {frase}")
print(f"🔢 IDs correspondentes: {ids_frase}")

# Hiperparâmetros
tamanho_vocab = len(vocabulario)
d_modelo = 64          # paper original usa 512; reduzido para CPU
d_ff = d_modelo * 4    # dimensão interna da FFN = 256
N_CAMADAS = 6
tamanho_lote = 1       # batch size

# Tabela de embeddings aleatória: shape (vocab, d_modelo)
tabela_embeddings = np.random.randn(tamanho_vocab, d_modelo)

# Tensor de entrada X: shape (lote, comprimento_sequencia, d_modelo)
X = tabela_embeddings[ids_frase]      # (SeqLen, d_modelo)
X = X[np.newaxis, :, :]              # (1, SeqLen, d_modelo)
print(f"\n✅ Shape do tensor de entrada X: {X.shape}")


# ════════════════════════════════════════════════
# PASSO 2: MOTOR MATEMÁTICO
# ════════════════════════════════════════════════

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax numericamente estável aplicado no último eixo.
    Subtrai o máximo para evitar overflow em np.exp.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class AtencaoEscalarPontual:
    """
    Scaled Dot-Product Attention:
        Atenção(Q, K, V) = softmax( Q·Kᵀ / sqrt(dk) ) · V

    Etapas:
        1. Inicializa matrizes de peso WQ, WK, WV aleatórias.
        2. Projeta X em Q, K e V.
        3. Calcula produto escalar Q·Kᵀ.
        4. Aplica scaling por sqrt(dk).
        5. Aplica softmax (implementação própria).
        6. Multiplica pelos valores V.
    """

    def __init__(self, d_modelo: int):
        self.dk = d_modelo
        self.WQ = np.random.randn(d_modelo, d_modelo)
        self.WK = np.random.randn(d_modelo, d_modelo)
        self.WV = np.random.randn(d_modelo, d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:
        Q = X @ self.WQ                               # (B, T, d_modelo)
        K = X @ self.WK
        V = X @ self.WV

        pontuacoes = Q @ K.transpose(0, 2, 1) / np.sqrt(self.dk)  # (B, T, T)
        pesos = softmax(pontuacoes)                                 # (B, T, T)
        return pesos @ V                                            # (B, T, d_modelo)


class NormalizacaoCamada:
    """
    Layer Normalization — normaliza ao longo dos features (último eixo).

    Etapas:
        1. Calcula média e variância por token.
        2. Normaliza: (X - média) / sqrt(variância + epsilon).
        3. Aplica parâmetros treináveis gamma (escala) e beta (deslocamento).
    """

    def __init__(self, d_modelo: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.gamma = np.ones(d_modelo)   # escala
        self.beta = np.zeros(d_modelo)   # deslocamento

    def propagar(self, X: np.ndarray) -> np.ndarray:
        media = np.mean(X, axis=-1, keepdims=True)
        variancia = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - media) / np.sqrt(variancia + self.epsilon)
        return self.gamma * X_norm + self.beta


class RedeFeedForward:
    """
    Feed-Forward Network:
        FFN(x) = max(0, x·W1 + b1)·W2 + b2

    Etapas:
        1. Transformação linear W1: expande d_modelo → d_ff.
        2. Ativação ReLU via np.maximum(0, x).
        3. Transformação linear W2: contrai d_ff → d_modelo.
    """

    def __init__(self, d_modelo: int, d_ff: int):
        self.W1 = np.random.randn(d_modelo, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_modelo)
        self.b2 = np.zeros(d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:
        oculto = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        return oculto @ self.W2 + self.b2


class CamadaEncoder:
    """
    Camada completa do Encoder (fluxo exato conforme o laboratório):
        1. X_att   = AtencaoEscalarPontual(X)
        2. X_norm1 = LayerNorm(X + X_att)       ← Add & Norm
        3. X_ffn   = FFN(X_norm1)
        4. X_saida = LayerNorm(X_norm1 + X_ffn)  ← Add & Norm
    """

    def __init__(self, d_modelo: int, d_ff: int):
        self.atencao = AtencaoEscalarPontual(d_modelo)
        self.norm1 = NormalizacaoCamada(d_modelo)
        self.ffn = RedeFeedForward(d_modelo, d_ff)
        self.norm2 = NormalizacaoCamada(d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:
        # Sub-camada 1: Auto-atenção + conexão residual
        X_att = self.atencao.propagar(X)
        X_norm1 = self.norm1.propagar(X + X_att)

        # Sub-camada 2: FFN + conexão residual
        X_ffn = self.ffn.propagar(X_norm1)
        X_saida = self.norm2.propagar(X_norm1 + X_ffn)

        return X_saida


# ════════════════════════════════════════════════
# PASSO 3: EMPILHANDO N=6 CAMADAS
# ════════════════════════════════════════════════

print("\n🔁 Passando o tensor pelas 6 camadas do Encoder...\n")

pilha_encoder = [CamadaEncoder(d_modelo, d_ff) for _ in range(N_CAMADAS)]

Z = X
for i, camada in enumerate(pilha_encoder):
    Z = camada.propagar(Z)
    print(f"  Camada {i + 1} → shape: {Z.shape}")

print(f"\n✅ Validação de sanidade: shape de entrada = {X.shape} | shape de saída = {Z.shape}")
print(f"   (esperado: (1, {len(frase)}, {d_modelo}))")
print("\n🧠 Vetor Z contextualizado — primeiros 8 valores do token 'o':")
print("  ", Z[0, 0, :8], "...")
