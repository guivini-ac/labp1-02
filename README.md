# Transformer Encoder — From Scratch

> **Disciplina:** Tópicos em Inteligência Artificial — 2026.1  
> **Professor:** Prof. Dimmy Magalhães  
> **Instituição:** iCEV - Instituto de Ensino Superior  
> **Laboratório:** P1-02

## Sobre o Projeto

Implementação completa do **Forward Pass** de um bloco Encoder do Transformer, conforme o artigo ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), utilizando **apenas** `Python`, `numpy` e `pandas`

## Arquitetura

```
Entrada (frase em texto)
        ↓
Tokenização + Embedding  →  X: (Lote, CompSeq, d_modelo)
        ↓
┌─── CamadaEncoder × 6 ──────────────────────────────────┐
│  1. AtencaoEscalarPontual(X)       → X_att              │
│  2. NormalizacaoCamada(X + X_att)  → X_norm1  (Add&Norm)│
│  3. RedeFeedForward(X_norm1)       → X_ffn              │
│  4. NormalizacaoCamada(X_norm1 + X_ffn) → X_saida       │
└────────────────────────────────────────────────────────-┘
        ↓
  Z: Representação densa contextualizada
```

## Pré-requisitos

```bash
pip install numpy pandas
```

## Como Executar

```bash
git clone https://github.com/guivini-ac/labp1-02.git
cd labp1-02
python transformer.py
```

## Saída Esperada

```
 Vocabulário carregado:
  palavra  id
        o   0
    banco   1
 bloqueou   2
   cartao   3
      meu   4
   ontem    5

 Frase de entrada : ['o', 'banco', 'bloqueou', 'meu', 'cartao']
 IDs correspondentes: [0, 1, 2, 4, 3]

 Shape do tensor de entrada X: (1, 5, 64)

 Passando o tensor pelas 6 camadas do Encoder...

  Camada 1 → shape: (1, 5, 64)
  Camada 2 → shape: (1, 5, 64)
  Camada 3 → shape: (1, 5, 64)
  Camada 4 → shape: (1, 5, 64)
  Camada 5 → shape: (1, 5, 64)
  Camada 6 → shape: (1, 5, 64)

 Validação de sanidade: shape de entrada = (1, 5, 64) | shape de saída = (1, 5, 64)
```

## Hiperparâmetros

| Parâmetro  | Valor usado | Valor original (paper) |
|------------|-------------|------------------------|
| `d_modelo` | 64          | 512                    |
| `d_ff`     | 256         | 2048                   |
| `N`        | 6           | 6                      |
| `heads`    | 1           | 8                      |

## Estrutura do Repositório

```
.
├── transformer.py   # Implementação completa
└── README.md        # Este arquivo
```

## Nota de Integridade Acadêmica

Ferramentas de IA Generativa foram consultadas como apoio para revisão de sintaxe do `numpy` e verificação de boas práticas, conforme autorizado pelo Contrato Pedagógico da disciplina. Toda a lógica matemática e estrutura do código foram desenvolvidas com base no artigo original.
