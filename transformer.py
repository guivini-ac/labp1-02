import numpy as np
import pandas as pd

np.random.seed(42)

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


frase = ["o", "banco", "bloqueou", "meu", "cartao"]
ids_frase = [vocabulario[palavra] for palavra in frase]
print(f"\n Frase de entrada : {frase}")
print(f" IDs correspondentes: {ids_frase}")


tamanho_vocab = len(vocabulario)
d_modelo = 64         
d_ff = d_modelo * 4   
N_CAMADAS = 6
tamanho_lote = 1     


tabela_embeddings = np.random.randn(tamanho_vocab, d_modelo)


X = tabela_embeddings[ids_frase]      
X = X[np.newaxis, :, :]            
print(f"\n Shape do tensor de entrada X: {X.shape}")



def softmax(x: np.ndarray) -> np.ndarray:

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class AtencaoEscalarPontual:


    def __init__(self, d_modelo: int):
        self.dk = d_modelo
        self.WQ = np.random.randn(d_modelo, d_modelo)
        self.WK = np.random.randn(d_modelo, d_modelo)
        self.WV = np.random.randn(d_modelo, d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:
        Q = X @ self.WQ                              
        K = X @ self.WK
        V = X @ self.WV

        pontuacoes = Q @ K.transpose(0, 2, 1) / np.sqrt(self.dk)  
        pesos = softmax(pontuacoes)                                
        return pesos @ V                                            


class NormalizacaoCamada:


    def __init__(self, d_modelo: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.gamma = np.ones(d_modelo)  
        self.beta = np.zeros(d_modelo)  

    def propagar(self, X: np.ndarray) -> np.ndarray:
        media = np.mean(X, axis=-1, keepdims=True)
        variancia = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - media) / np.sqrt(variancia + self.epsilon)
        return self.gamma * X_norm + self.beta


class RedeFeedForward:

    def __init__(self, d_modelo: int, d_ff: int):
        self.W1 = np.random.randn(d_modelo, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_modelo)
        self.b2 = np.zeros(d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:
        oculto = np.maximum(0, X @ self.W1 + self.b1) 
        return oculto @ self.W2 + self.b2


class CamadaEncoder:


    def __init__(self, d_modelo: int, d_ff: int):
        self.atencao = AtencaoEscalarPontual(d_modelo)
        self.norm1 = NormalizacaoCamada(d_modelo)
        self.ffn = RedeFeedForward(d_modelo, d_ff)
        self.norm2 = NormalizacaoCamada(d_modelo)

    def propagar(self, X: np.ndarray) -> np.ndarray:

        X_att = self.atencao.propagar(X)
        X_norm1 = self.norm1.propagar(X + X_att)


        X_ffn = self.ffn.propagar(X_norm1)
        X_saida = self.norm2.propagar(X_norm1 + X_ffn)

        return X_saida



print("\n Passando o tensor pelas 6 camadas do Encoder...\n")

pilha_encoder = [CamadaEncoder(d_modelo, d_ff) for _ in range(N_CAMADAS)]

Z = X
for i, camada in enumerate(pilha_encoder):
    Z = camada.propagar(Z)
    print(f"  Camada {i + 1} → shape: {Z.shape}")

print(f"\n Validação de sanidade: shape de entrada = {X.shape} | shape de saída = {Z.shape}")
print(f"   (esperado: (1, {len(frase)}, {d_modelo}))")
print("\n Vetor Z contextualizado — primeiros 8 valores do token 'o':")
print("  ", Z[0, 0, :8], "...")
