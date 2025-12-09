---
title: "QR Decomposition"
categories: [math]
tags: [linear-algebra, qr, matrix-factorization]
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


# 1. QR Decomposition이란?

QR 분해는 행렬 $A \in \mathbb{R}^{m \times n}$ (보통 $m \ge n$)를  
두 개의 행렬 **Q**, **R**의 곱으로 나타내는 분해다:

$$
A = QR
$$

- $Q \in \mathbb{R}^{m \times m}$ : **직교(orthogonal)** matrix  
  → $Q^T Q = I$, 열벡터들이 서로 직교하고 길이가 1 → orthonormal basis  
- $R \in \mathbb{R}^{m \times n}$ : **upper triangular** matrix  
  → 아래쪽이 모두 0 (조금 더 정확히 말하자면, $t_{ij}$가 R 행렬의 entry 일 때, $t_{ij}=0$ for $i>j$)

QR 분해는 *“복잡한 행렬을 정리된 기저(Q)와 구조적 계수(R)로 바꿔놓는 과정”*이라고 이해하면 된다.

---

# 2. 왜 Q가 orthonormal이고 R이 upper triangular가 되는가?

## 2.1 직교 기저로 바꾼다는 의미

행렬 A를 열벡터로 표현하면:

$$
A = [a_1, a_2, \dots, a_n]
$$

이 벡터들을 **직교화(orthogonalize)** 해서 새로운 벡터들을 만든다:

$$
q_1, q_2, \dots, q_n
$$

그람–슈미트(GS)를 적용하면:

$$
a_j = r_{1j} q_1 + r_{2j} q_2 + \dots + r_{jj} q_j
$$

여기서 중요한 점:

- $a_j$ 는 **오직 $q_1, \dots, q_j$**까지만 필요  
- $q_{j+1}$ 이상의 방향은 필요 없다 → 계수가 0  
- 그래서 R이 **upper triangular matrix 구조**를 가진다

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/769ee0585bb75d9745c9ceecec4ecfdf9488195b" 
     alt="QR decomposition formula" 
     width="300">

---

# 3. Gram–Schmidt 과정과 QR의 관계

QR decomposition을 잘 이해하기 위해서는 Gram–Schmidt process(GS)에 대해서 이해하고 있어야 한다.  
Gram–Schmidt process(GS)는 "orthonormal basis를 만드는 알고리즘"이고  
QR 분해는 이 과정을 **행렬 형태로 깔끔하게 정리한 것**이다.

정확히는:

1. Q의 열벡터는 GS로 얻은 orthonormal basis  
2. R의 원소는 각 $a_j$ 가 $q_i$ 방향으로 얼마만큼 투영되는지 나타낸 계수

즉,

$$
r_{ij} = q_i^T a_j
$$

이 관계식 하나가 Q와 R의 역할을 정확하게 설명한다.

---

# 4. QR 분해의 직관적 의미

### 🔹 Q = 방향 정보 (정돈된 축)  
복잡한 열벡터들을 **서로 직교하고 길이가 1인 축**으로 다시 정렬한 것.

### 🔹 R = 크기·관계 정보  
원래 벡터들이 그 축 위에서 갖는 계수들.  
이 계수가 위로 몰리기 때문에 상삼각 구조가 된다.

그래서 QR 분해는  
**"벡터 공간을 직교 기저로 재정렬하고, 원래 벡터들이 그 기저 위에서 어떻게 표현되는지 정리하는 과정"**이다.

---

# 5. 수학적 성질 정리

### 1) 직교 행렬 Q
- 길이 보존: $\|Qx\| = \|x\|$  
- 내적 보존: $(Qx) \cdot (Qy) = x \cdot y$  
- 기하적으로 **회전 + 반사(reflection)** 변환에 해당

### 2) R의 상삼각 구조
- A의 열벡터들이 순차적으로 확장되는 subspace 구조 때문  
- column space의 계층적 구조가 R에 그대로 반영됨

### 3) QR 분해는 항상 존재  
- “full rank”인 경우 유일  
- Householder, Givens 방식으로 더 안정적인 QR도 존재

---

# 6. QR이 어디서 응용되는가? (매우 중요함)

QR 분해는 수학 이론뿐 아니라 **실제 엔지니어링, 머신러닝, 최적화 분야에서 핵심적인 역할**을 한다.

## ⭐ 1) 선형회귀(Least Squares) — 가장 많이 쓰임

선형회귀 문제 $Ax = b$는 보통 정규방정식으로 풀지만:

$$
A^T A x = A^T b
$$

이 방식은 수치적으로 불안정하다.

QR을 이용하면:

$$
A = QR \Rightarrow Rx = Q^T b
$$

R이 upper triangular matrix이기 때문에 back-substitution으로 빠르고 안정적으로 풀 수 있다.

---

## ⭐ 2) 정사영(projection) 계산

벡터 $b$를 A의 column space에 정사영하려면:

$$
\text{proj}_A(b) = QQ^T b
$$

Q가 직교이기 때문에 projection이 매우 간단해진다.

---

## ⭐ 3) SVD 계산의 전처리 (bidiagonalization)

SVD는 QR을 반복적으로 적용하는 Householder reduction 기반으로 돌아간다.

---

## ⭐ 4) Eigenvalue 계산 (QR Algorithm)

1. 행렬 A를 QR로 분해하고  
2. $A_{k+1} = R_k Q_k$ 로 갱신  
3. 반복하면 고유값으로 수렴  

➡ MATLAB, NumPy, LAPACK 내부에서 사용되는 표준 알고리즘

---

## ⭐ 5) 머신러닝 최적화

- Normal Equation 대체  
- Orthogonalization 필요한 경우  
- PCA 준비 과정  
- compressed sensing 복원 등

---

# 7. 정리

- QR은 벡터 공간을 **정돈된 직교 기저(Q)**로 바꾸고  
  그 위에서 원래 벡터의 **계수(R)**를 얻는 과정이다.
- 머신러닝·수치해석·최적화에서 QR은 **안정성의 핵심 도구**이다.
- Householder QR과 QR 알고리즘은 실제 구현에서 매우 중요하다.

