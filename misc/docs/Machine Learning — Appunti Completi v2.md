# 📘 Machine Learning --- Appunti Completi v2

> **Corso:** Machine Learning --- Claudia d'Amato, Nicola Di Mauro\
> **Dipartimento:** Informatica, Università di Bari

> **Nota editoriale:** questa versione riorganizza gli appunti originali
> per renderli più navigabili e adatti allo studio. I contenuti teorici
> sono mantenuti; gli interventi principali riguardano struttura,
> gerarchia, indice e separazione delle informazioni organizzative.

------------------------------------------------------------------------

## 🧭 Come sono organizzati gli appunti

-   **Parte I:** basi del ML, regressione e classificazione.
-   **Parte II:** valutazione dei modelli, metriche, costi e WEKA.
-   **Parte III:** probabilistic graphical models e inferenza.
-   **Parte IV:** alberi di decisione, ensemble learning e kernel/SVM.
-   **Parte V:** deep learning e tractable probabilistic models.
-   **Parte VI:** NLP con deep learning, attention e Transformer.
-   **Parte VII:** reinforcement learning.
-   **Parte VIII:** ILP, knowledge graph embeddings e clustering.

### 🏷️ Chiavi di lettura

-   **Concetto:** definizione e intuizione.
-   **Modello:** formulazione matematica.
-   **Algoritmo:** procedura di apprendimento/inferenza.
-   **Valutazione:** metriche e protocolli sperimentali.
-   **Confronto:** differenze, vantaggi e limiti tra metodi.

------------------------------------------------------------------------

## 📑 Indice

### Informazioni del corso

-   [Informazioni organizzative](#-informazioni-organizzative)
-   **PARTE I --- Fondamenti del Machine Learning**
    -   [01](#01)
    -   [02 --- Concetti Fondamentali](#02--concetti-fondamentali)
-   **PARTE II --- Valutazione, metriche e strumenti**
    -   [03 --- Valutazione dei Modelli](#03--valutazione-dei-modelli)
    -   [04 --- Intervalli di Confidenza e Test
        Statistici](#04--intervalli-di-confidenza-e-test-statistici)
    -   [05 --- Predizione di Probabilità e
        Costi](#05--predizione-di-probabilit-e-costi)
    -   [06 --- Valutazione della Predizione
        Numerica](#06--valutazione-della-predizione-numerica)
    -   [07 --- WEKA](#07--weka)
-   **PARTE III --- Modelli probabilistici e grafici**
    -   [08 --- Probabilistic Graphical Models
        (PGM)](#08--probabilistic-graphical-models-pgm)
-   **PARTE IV --- Alberi, ensemble e kernel methods**
    -   [09 --- Alberi di Decisione e
        Regole](#09--alberi-di-decisione-e-regole)
    -   [13 --- Boosting e Ensemble
        Learning](#13--boosting-e-ensemble-learning)
    -   [10 --- Kernel e Support Vector Machines
        (SVM)](#10--kernel-e-support-vector-machines-svm)
-   **PARTE V --- Deep Learning e modelli generativi**
    -   [11 --- Deep Learning](#11--deep-learning)
    -   [12 --- Tractable Probabilistic Models
        (TPM)](#12--tractable-probabilistic-models-tpm)
-   **PARTE VI --- Natural Language Processing**
    -   [14 --- Natural Language Processing con Deep
        Learning](#14--natural-language-processing-con-deep-learning)
-   **PARTE VII --- Reinforcement Learning**
    -   [15 --- Reinforcement Learning
        (RL)](#15--reinforcement-learning-rl)
-   **PARTE VIII --- Apprendimento relazionale e rappresentazioni**
    -   [16 --- Apprendimento Relazionale e Inductive Logic Programming
        (ILP)](#16--apprendimento-relazionale-e-inductive-logic-programming-ilp)
    -   [17 --- Knowledge Graph Embeddings
        (KGE)](#17--knowledge-graph-embeddings-kge)
    -   [18 --- Clustering Basato su
        Distanza](#18--clustering-basato-su-distanza)

------------------------------------------------------------------------

# 🧑‍🏫 Informazioni organizzative

# 01 --- Introduzione al Machine Learning

> *Riferimento: Murphy, Machine Learning: a Probabilistic Perspective,
> Ch. 1*

------------------------------------------------------------------------

## 👤 Docenti

  -------------------------------------------------------------------------------------
  Docente           Ufficio           Ricevimento             Email
  ----------------- ----------------- ----------------------- -------------------------
  Claudia d'Amato   520 @ DIB         Martedì 15:00--16:30 /  claudia.damato@uniba.it
                                      MS Teams (codice:       
                                      `2cnon2n`)              

  Nicola Di Mauro   513 @ DIB         Su appuntamento         nicola.dimauro@uniba.it
  -------------------------------------------------------------------------------------

**Piattaforma e-learning:**
https://elearning.uniba.it/course/view.php?id=4893\
**Password:** `MLCS-2425`

------------------------------------------------------------------------

## 📚 Prerequisiti

-   **Algebra lineare:** manipolazione di vettori e matrici
-   **Calcolo:** derivate parziali
-   **Probabilità:** distribuzioni comuni, Regola di Bayes
-   **Statistica:** media/mediana/moda, massima verosimiglianza
-   **Riferimento:** Sheldon Ross, *A First Course in Probability*

------------------------------------------------------------------------

## 📖 Libri di Testo

  ------------------------------------------------------------------------
  Sigla            Libro            Autore/i                Anno
  ---------------- ---------------- ----------------------- --------------
  MLPP             Machine          Murphy                  2012
                   Learning: a                              
                   Probabilistic                            
                   Perspective                              

  PRML             Pattern          Bishop                  2018
                   Recognition and                          
                   Machine Learning                         

  LRL              Logical and      De Raedt                2008
                   Relational                               
                   Learning                                 

  DM               Data Mining:     Witten & Frank          2017
                   Practical ML                             
                   Tools and                                
                   Techniques                               

  DL               Deep Learning    Goodfellow, Bengio,     2017
                                    Courville               

  GDL              Grokking Deep    Trask                   2019
                   Learning                                 

  DLP              Deep Learning    Chollet                 2018
                   with Python                              
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Software di Riferimento

  Tool           URL                                     Descrizione
  -------------- --------------------------------------- --------------------------
  WEKA           https://www.cs.waikato.ac.nz/ml/weka/   Workbench per ML
  scikit-learn   https://scikit-learn.org/stable/        ML con Python
  Keras          https://keras.io/                       Deep Learning con Python
  PyTorch        https://pytorch.org/                    Framework per ML

------------------------------------------------------------------------

## 🎯 Valutazione del Corso

### Progetto

-   Applicare ML a un problema reale (scelto o assegnato)
-   Gruppi di **max 2 studenti**
-   Discussione durante l'esame (uso di slide consigliato)
-   Report scritto da inviare **2--3 giorni prima** dell'esame

### Esame Orale

-   Domande teoriche generali sul corso
-   Discussione del progetto
-   Durata: **20--25 minuti**

------------------------------------------------------------------------

------------------------------------------------------------------------

# PARTE I --- Fondamenti del Machine Learning

## 🧠 Cos'è il Machine Learning?

``` mermaid
flowchart LR
    A[Problema] --> B{Regole note?}
    B -->|Sì| C[Programma tradizionale]
    B -->|No| D[Raccolta esempi]
    D --> E[Algoritmo di ML]
    E --> F[Modello appreso]
    F --> G[Predizione su nuovi casi]
```

-   Come informatici, scriviamo programmi che codificano regole per
    risolvere un problema
-   In molti casi è **molto difficile** specificare queste regole (es.
    riconoscere un gatto in un'immagine)
-   I sistemi di apprendimento **non sono programmati direttamente** per
    risolvere il problema, ma sviluppano il proprio programma basandosi
    su:
    -   **Esempi** di come dovrebbero comportarsi
    -   **Esperienza per tentativi** (trial-and-error)
-   L'apprendimento significa **incorporare informazioni** dagli esempi
    di training nel sistema

------------------------------------------------------------------------

## 🤔 Perché Usare l'Apprendimento?

-   È molto difficile scrivere programmi per problemi come riconoscere
    cifre scritte a mano
    -   Cosa distingue un 2 da un 7? Come funziona il nostro cervello?
-   Invece di scrivere un programma a mano, **raccogliamo esempi** che
    specificano l'output corretto per un dato input
-   L'algoritmo di ML produce un programma che fa il lavoro --- può
    contenere **milioni di parametri**
-   Se fatto correttamente, il programma funziona anche per **nuovi
    casi** (generalizzazione)

------------------------------------------------------------------------

## 🌍 Applicazioni del ML

``` mermaid
mindmap
  root((Applicazioni ML))
    Classificazione
      Riconoscimento vocale
      Identità facciale
      Diagnosi medica
    Raccomandazione
      Amazon
      Netflix
    Information Retrieval
      Ricerca documenti
      Ricerca immagini
    Visione Artificiale
      Detection
      Segmentazione
      Stima profondità
    Robotica
      Percezione
      Pianificazione
    Giochi
      Imparare a giocare
    Anomalie
      Transazioni fraudolente
      Situazioni di panico
    Filtraggio
      Spam
      Frodi
```

------------------------------------------------------------------------

## 🔄 Paradigmi del Machine Learning

``` mermaid
flowchart TB
    subgraph Supervised["🔍 Supervised Learning"]
        direction TB
        S1[Output corretto noto per ogni esempio]
        S2[Classificazione: output discreto]
        S3[Regressione: output continuo]
        S1 --> S2
        S1 --> S3
    end

    subgraph Unsupervised["🔎 Unsupervised Learning"]
        direction TB
        U1[Nessun output noto]
        U2[Scoprire pattern/struttura]
        U3[Clustering, estrazione feature]
        U1 --> U2
        U2 --> U3
    end

    subgraph Reinforcement["🎮 Reinforcement Learning"]
        direction TB
        R1[Azione in ambiente dinamico]
        R2[Massimizzare ricompensa]
        R3[Ricompensa spesso ritardata]
        R1 --> R2
        R2 --> R3
    end

    Supervised ~~~ Unsupervised ~~~ Reinforcement
```

### Supervised Learning

-   Output corretto noto per ogni esempio di training
-   **Classificazione:** output discreto (riconoscimento vocale,
    diagnosi medica)
-   **Regressione:** output continuo (predizione prezzi, rating)

### Unsupervised Learning

-   Nessun output desiderato --- si cerca struttura nei dati
-   Scoprire pattern: correlazioni, cluster, trend, anomalie
-   Domanda chiave: *come sappiamo se una rappresentazione è buona?*

### Reinforcement Learning

-   Un agente intelligente prende azioni in un ambiente dinamico
-   Obiettivo: massimizzare un segnale di ricompensa
-   La ricompensa è spesso **ritardata** e contiene poca informazione

------------------------------------------------------------------------

## � Protocolli di Apprendimento

Oltre ai tre paradigmi principali, esistono diversi **protocolli** che
specificano come i dati vengono usati durante l'apprendimento.

### Batch vs Online Learning

  -------------------------------------------------------------------------------------
                      Batch Learning                  Online Learning
  ------------------- ------------------------------- ---------------------------------
  **Dati**            Tutto il dataset disponibile in I dati arrivano in streaming
                      anticipo                        

  **Aggiornamento**   Dopo aver visto tutti i dati    Dopo ogni esempio (o mini-batch)

  **Uso**             Dataset statici                 Dati in continuo arrivo (es. log,
                                                      sensori)

  **Esempio**         Addestrare un modello su un     Adattare un modello a nuovi dati
                      dataset completo                in tempo reale
  -------------------------------------------------------------------------------------

### Semi-Supervised Learning

-   **Situazione:** abbiamo pochi dati **etichettati** e molti dati
    **non etichettati**
-   **Idea:** usare i dati non etichettati per migliorare
    l'apprendimento (es. capire la struttura dei dati)
-   **Motivazione:** etichettare i dati è costoso, mentre i dati non
    etichettati sono abbondanti
-   **Esempio:** classificazione di documenti dove solo alcuni sono
    etichettati

### Transductive Learning

-   **Situazione:** abbiamo dati etichettati e un **insieme specifico di
    dati di test** non etichettati
-   **Obiettivo:** predire le etichette **solo** per quei dati di test
    specifici
-   **Differenza dall'induttivo:** non si costruisce una funzione
    generale $f$, ma si sfrutta la struttura dei dati di test stessi
-   **Esempio:** classificare un insieme fisso di documenti non
    etichettati

``` mermaid
flowchart TB
    A[Protocolli di Apprendimento] --> B[Batch<br/>tutti i dati in anticipo]
    A --> C[Online<br/>dati in streaming]
    A --> D[Semi-supervised<br/>pochi etichettati + molti non etichettati]
    A --> E[Transductive<br/>predire solo su test set specifico]
```

------------------------------------------------------------------------

## �📊 Tipi di Task di ML

### Task Predittivi (Supervised Learning)

-   Training set: $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$
-   Obiettivo: imparare una mappatura $\mathbf{x} \to y$
-   $\mathbf{x}_i$ può essere un vettore di feature o un oggetto
    strutturato (immagine, frase, ...)
-   $y_i$ può essere:
    -   **Categorico** ($y_i \in \{1, \ldots, C\}$) →
        **Classificazione**
    -   **Reale** ($y_i \in \mathbb{R}$) → **Regressione**
    -   **Ordinale** (es. voti A--F) → **Regressione ordinale**

### Task Descrittivi (Unsupervised Learning)

-   Solo input: $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^{N}$
-   Obiettivo: trovare pattern "interessanti" nei dati
-   Problema meno ben definito: nessuna metrica di errore ovvia

  ------------------------------------------------------------------------
  Tipo             Obiettivo                      Esempio
  ---------------- ------------------------------ ------------------------
  **Association    Scoprire relazioni tra         Market Basket Analysis
  Analysis**       variabili                      

  **Cluster        Raggruppare osservazioni       Raggruppare documenti
  Analysis**       simili                         per argomento

  **Anomaly        Identificare                   Rilevare frodi su carte
  Detection**      outlier/deviazioni             di credito
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## ⚖️ ML vs Data Mining vs Statistica

``` mermaid
flowchart LR
    subgraph DM["Data Mining"]
        D1[Tecniche ML semplici]
        D2[Database molto grandi]
        D3[Velocità prioritaria]
    end
    subgraph ML["Machine Learning"]
        M1[Algoritmi complessi]
        M2[Problemi con sapore AI]
        M3[Demo impressionanti]
    end
    subgraph ST["Statistica"]
        S1[Teoria rigorosa]
        S2[Stima asintotica]
        S3[Procedure semplici]
    end
    
    DM <--> ML <--> ST
```

  ---------------------------------------------------------------------------
  Aspetto         Data Mining       Machine Learning         Statistica
  --------------- ----------------- ------------------------ ----------------
  **Focus**       Database grandi   Problemi AI              Inferenza
                                                             rigorosa

  **Approccio**   Tecniche semplici Algoritmi complessi      Stima asintotica

  **Obiettivo**   Velocità          Risultati impressionanti Correttezza
                                                             teorica

  **Confini**     Sempre più        Usa teoria statistica    Va oltre i
                  sfumati                                    problemi tipici
  ---------------------------------------------------------------------------

> **Citazione:** *"All models are wrong, but some models are useful."*
> --- George Box, 1987

------------------------------------------------------------------------

## 🏷️ Classificazione

L'obiettivo è imparare una mappatura dagli input $\mathbf{x}$ agli
output $y$, dove $y \in \{1, \dots, C\}$, con $C$ il numero di classi:

-   Se $C = 2$ → **classificazione binaria**
-   Se $C > 2$ → **classificazione multiclasse**
-   Se le etichette di classe non sono mutuamente esclusive →
    **classificazione multi-label**

**Formalizzazione come approssimazione di funzione:** - Assumiamo
$y = f(\mathbf{x})$ per qualche funzione sconosciuta $f$ - L'obiettivo
dell'apprendimento è stimare la funzione $f$ dato un training set
etichettato, e poi fare predizioni usando
$\hat{y} = \hat{f}(\mathbf{x})$

**Obiettivo principale:** fare predizioni su input **nuovi** (mai visti
prima) --- **generalizzazione** --- poiché predire la risposta sul
training set è facile.

### Predizioni Probabilistiche

Per gestire casi ambigui è desiderabile restituire una **probabilità**:

-   Denotiamo la distribuzione di probabilità sulle etichette possibili,
    dato l'input $\mathbf{x}$ e il training set $\mathcal{D}$, con
    $p(y \mid \mathbf{x}, \mathcal{D})$
-   Se ci sono solo due classi, è sufficiente restituire il singolo
    numero $p(y = 1 \mid \mathbf{x}, \mathcal{D})$, poiché
    $p(y = 1 \mid \mathbf{x}, \mathcal{D}) + p(y = 0 \mid \mathbf{x}, \mathcal{D}) = 1$

Data un'output probabilistico, possiamo sempre calcolare la nostra
"migliore ipotesi" della "vera etichetta" usando:

$$\hat{y} = \hat{f}(\mathbf{x}) = \arg\max_{c=1}^{C} p(y = c \mid \mathbf{x}, \mathcal{D})$$

------------------------------------------------------------------------

## 🔎 Unsupervised Learning: Dettagli

L'obiettivo è scoprire "struttura interessante" nei dati: - A differenza
del supervised learning, non ci viene detto qual è l'output desiderato
per ogni input - Formalizziamo il task come **density estimation** →
costruiamo modelli della forma $p(\mathbf{x}_i \mid \theta)$

**Due differenze dal caso supervised:** 1. Scriviamo
$p(\mathbf{x}_i \mid \theta)$ invece di
$p(y_i \mid \mathbf{x}_i, \theta)$ - Il supervised learning è
**conditional density estimation**, mentre l'unsupervised è
**unconditional density estimation** 2. $\mathbf{x}_i$ è un vettore di
feature, quindi dobbiamo creare **modelli di probabilità
multivariati** - Nel supervised learning, $y_i$ è di solito una singola
variabile che cerchiamo di predire - Questo significa che per la maggior
parte dei problemi supervised possiamo usare modelli di probabilità
**univariati**, semplificando significativamente il problema

### Scoprire Cluster

Un esempio canonico di unsupervised learning: **clustering** dei dati in
gruppi.

Sia $K$ il numero di cluster: - **Primo obiettivo:** stimare la
distribuzione sul numero di cluster, $p(K \mid \mathcal{D})$ - **Secondo
obiettivo:** stimare a quale cluster appartiene ogni punto - Spesso
approssimiamo la distribuzione $p(K \mid \mathcal{D})$ con la sua moda,
$K^* = \arg\max_K p(K \mid \mathcal{D})$ - Sia $z_i \in \{1, \dots, K\}$
il cluster a cui il punto dati $i$ è assegnato - $z_i$ è un esempio di
**variabile nascosta o latente**, poiché non è mai osservata nel
training set - Possiamo inferire a quale cluster appartiene ogni punto
dati calcolando
$z_i^* = \arg\max_k p(z_i = k \mid \mathbf{x}_i, \mathcal{D})$

### Scoprire Fattori Latenti

Quando si trattano dati ad alta dimensionalità, è spesso utile ridurre
la dimensionalità proiettando i dati su un sottospazio a dimensione
inferiore che cattura l'"essenza" dei dati --- **dimensionality
reduction**:

-   **Motivazione:** sebbene i dati possano apparire ad alta
    dimensionalità, potrebbero esserci solo un piccolo numero di gradi
    di variabilità, corrispondenti a **fattori latenti**
-   Es. modellando l'apparenza di immagini di volti, potrebbero esserci
    solo pochi fattori latenti sottostanti che descrivono la maggior
    parte della variabilità, come illuminazione, posa, identità, ...
-   Quando usate come input ad altri modelli statistici, tali
    rappresentazioni a bassa dimensione spesso risultano in **migliore
    accuratezza predittiva**, perché si concentrano sull'"essenza"
    dell'oggetto, filtrando le feature inessenziali
-   L'approccio più comune alla riduzione della dimensionalità è
    chiamato **principal components analysis (PCA)**
-   Il modello ha la forma $z \to y$
-   Questo può essere pensato come una versione unsupervised della
    regressione lineare (multi-output), dove osserviamo la risposta ad
    alta dimensione $y$, ma non la "causa" a bassa dimensione $z$
-   Dobbiamo "invertire la freccia", e inferire il $z$ latente a bassa
    dimensione dal $y$ osservato ad alta dimensione

------------------------------------------------------------------------

## 🎯 No Free Lunch Theorem

> *"All models are wrong, but some models are useful."* --- George Box,
> 1987

-   Gran parte del machine learning si occupa di ideare diversi modelli,
    e diversi algoritmi per adattarli

-   **Non esiste un modello universalmente migliore** --- questo è a
    volte chiamato il **no free lunch theorem**

-   ## La ragione è che un insieme di assunzioni che funziona bene in un dominio può funzionare male in un altro

```{=html}
<!-- ===== SEZIONE: 02_concetti_fondamentali.md ===== -->
```
# 02 --- Concetti Fondamentali

> *Riferimenti: Murphy, MLPP Ch. 1, 3.5 · Bishop, PRML Ch. 1, 3, 4*

------------------------------------------------------------------------

## 🧩 Modelli Parametrici vs Non-Parametrici

Nei modelli probabilistici ci concentriamo su forme del tipo:

$$p(y \mid x) \quad \text{(supervised)} \qquad \text{oppure} \qquad p(x) \quad \text{(unsupervised)}$$

``` mermaid
flowchart LR
    A[Modello] --> B{Numero di parametri?}
    B -->|Fisso| C[Parametrico]
    B -->|Cresce con i dati| D[Non-parametrico]
    C --> C1[Più veloce]
    C --> C2[Assunzioni più forti]
    D --> D1[Più flessibile]
    D --> D2[Spesso computazionalmente costoso]
```

  ---------------------------------------------------------------------------------
                   Parametrico                 Non-parametrico
  ---------------- --------------------------- ------------------------------------
  **Parametri**    Numero fisso                Crescono con i dati

  **Vantaggio**    Più veloce da usare         Più flessibile

  **Svantaggio**   Assunzioni più forti sulla  Spesso intrattabile per dataset
                   distribuzione               grandi

  **Esempio**      Regressione lineare, Naive  KNN, alberi di decisione
                   Bayes                       
  ---------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📏 Regressione Lineare

### Il Modello

-   **Output continuo** $t \in \mathbb{R}$
-   Modello lineare semplice:

$$y(x) = w_0 + w_1 x$$

-   **Loss (errore quadratico)** tra predizione $y$ e valore vero $t$:

$$\ell(w) = \sum_{n=1}^{N} \left[ t^{(n)} - (w_0 + w_1 x^{(n)}) \right]^2$$

> Geometricamente, la loss rappresenta la **distanza verticale** tra i
> punti dati e la retta del modello.

### Ottimizzazione: Gradient Descent

``` mermaid
flowchart LR
    A[Inizializza w] --> B[Calcola gradiente ∂ℓ/∂w]
    B --> C[Aggiorna w ← w − λ ∂ℓ/∂w]
    C --> D{Convergenza?}
    D -->|No| B
    D -->|Sì| E[Modello ottimizzato]
```

-   **λ** è il *learning rate*
-   Per un singolo caso di training → regola **LMS** (Least Mean
    Squares):

$$w \leftarrow w + 2\lambda \underbrace{(t^{(n)} - y(x^{(n)})) x^{(n)}}_{\text{errore}}$$

> Quando l'errore tende a zero, anche l'aggiornamento tende a zero (w
> smette di cambiare).

### Batch vs Stochastic Gradient Descent

  ------------------------------------------------------------------------------------
                      Batch GD                  Stochastic GD
  ------------------- ------------------------- --------------------------------------
  **Aggiornamento**   Su tutto il dataset       Su un singolo punto

  **Vantaggio**       Convergenza stabile       Gestisce la ridondanza, può uscire da
                                                minimi locali

  **Svantaggio**      Costoso, può fermarsi in  Non converge in modo netto
                      minimi locali             
  ------------------------------------------------------------------------------------

### Soluzione Analitica

Per la regressione lineare ai minimi quadrati esiste una soluzione in
forma chiusa:

$$w = (X^T X)^{-1} X^T t$$

> Si ottiene ponendo a zero le derivate della loss rispetto a $w$.

------------------------------------------------------------------------

## 📈 Fitting di un Polinomio

Se il modello lineare non è sufficiente, si possono usare **feature
combinate**:

$$y(x, w) = w_0 + \sum_{j=1}^{M} w_j x^j$$

-   $M$ = ordine del polinomio
-   Aumentando $M$ il modello diventa più complesso

``` mermaid
flowchart LR
    subgraph M0["M = 0"]
        A1[Costante]
    end
    subgraph M1["M = 1"]
        A2[Lineare]
    end
    subgraph M3["M = 3"]
        A3[Cubico]
    end
    subgraph M9["M = 9"]
        A4[Alto ordine]
    end
    M0 --> M1 --> M3 --> M9
```

------------------------------------------------------------------------

## ⚖️ Generalizzazione e Overfitting

-   **Generalizzazione** = capacità del modello di predire su dati non
    visti (held-out)
-   Un modello troppo complesso **memorizza** i dati di training
    (overfitting) ma fallisce sui nuovi dati

``` mermaid
flowchart TB
    A[Complessità del modello] --> B{Complessità adeguata?}
    B -->|Troppo semplice| C[Underfitting<br/>alto bias]
    B -->|Giusta| D[Buona generalizzazione]
    B -->|Troppo complesso| E[Overfitting<br/>alta varianza]
```

------------------------------------------------------------------------

## 🛡️ Regularized Least Squares (Ridge Regression)

Per selezionare automaticamente la complessità del modello si usa la
**regolarizzazione**:

$$\tilde{\ell}(w) = \sum_{n=1}^{N} \left[ t^{(n)} - (w_0 + w_1 x^{(n)}) \right]^2 + \alpha \sum_j w_j^2$$

-   Il secondo termine **penalizza** pesi grandi → incoraggia valori
    piccoli di $w$
-   Quando si penalizza il quadrato dei pesi → **ridge regression**
-   Il coefficiente $\alpha$ governa l'importanza relativa della
    regolarizzazione
-   **Risultato:** migliore generalizzazione (es. con $M=9$)

------------------------------------------------------------------------

## 🏷️ Classificazione

### Il Problema

-   Dati input $x$ (feature) e classi $y$ (label)
-   Esempio: altezza, peso, colore → elefante / orso

### Due Approcci

``` mermaid
flowchart TB
    subgraph Discriminative["🔴 Discriminativo"]
        D1[Impara mappatura diretta<br/>X → {0,1,...,K}]
        D2[Modella p(y|x) o il confine]
    end
    subgraph Generative["🟢 Generativo"]
        G1[Modella p(x|y)]
        G2[Classifica via Bayes]
    end
    Discriminative ~~~ Generative
```

                        Discriminativo                    Generativo
  --------------------- --------------------------------- ---------------------
  **Modella**           $p(y \mid x)$ o confine diretto   $p(x \mid y)$
  **Classificazione**   Diretta                           Via regola di Bayes
  **Esempio**           Regressione logistica, SVM        Naive Bayes

------------------------------------------------------------------------

## 📐 Classificazione come Regressione

-   **Trucco:** ignorare che l'output è categorico
-   Problema binario: $t \in \{-1, 1\}$
-   Modello: $y(x) = w^T x$
-   Loss quadratica:

$$\ell_{\text{square}}(w, t) = \frac{1}{N} \sum_{n=1}^{N} (t^{(n)} - w^T x^{(n)})^2$$

### Regola di Decisione

$$y = \begin{cases} 1 & \text{se } f(x, w) \geq 0 \\ -1 & \text{altrimenti} \end{cases}$$

------------------------------------------------------------------------

## 🎯 Algoritmi per Parametri Lineari

I classificatori lineari basati su soluzioni ai minimi quadrati
**mancano di robustezza agli outlier**: - La funzione di errore
somma-dei-quadrati penalizza predizioni "troppo corrette" (che giacciono
molto lontano sul lato corretto della decisione) - Oltre ai minimi
quadrati, altre alternative per apprendere i parametri delle funzioni
discriminanti lineari possono essere considerate

### Discriminante Lineare di Fisher

Il **discriminante lineare di Fisher** proietta i dati su una direzione
$w$ che **massimizza la separazione tra le classi**:

-   Massimizza il rapporto tra la varianza **tra le classi** e la
    varianza **entro le classi** (between-class / within-class)
-   La direzione ottimale è data da:

$$w \propto S_W^{-1} (\mu_1 - \mu_2)$$

dove $S_W$ è la matrice di scatter within-class e $\mu_1, \mu_2$ sono le
medie delle due classi.

### Algoritmo Perceptrone

Il **perceptron** (Rosenblatt, 1957) è un algoritmo iterativo per
apprendere i pesi di un classificatore lineare:

1.  Inizializza $w = 0$
2.  Per ogni esempio $(x_i, y_i)$:
    -   Se $y_i (w^T x_i) \leq 0$ (esempio misclassificato), aggiorna:
        $w \leftarrow w + y_i x_i$
3.  Ripeti finché tutti gli esempi sono classificati correttamente

-   **Converge** solo se i dati sono **linearmente separabili**
-   Non converge se i dati non sono separabili

``` mermaid
flowchart LR
    A[Inizializza w = 0] --> B[Per ogni esempio x_i]
    B --> C{Classificato correttamente?}
    C -->|No| D[Aggiorna w ← w + y_i x_i]
    C -->|Sì| E{Tutti corretti?}
    D --> B
    E -->|No| B
    E -->|Sì| F[Convergenza]
```

------------------------------------------------------------------------

## 💸 Funzioni di Loss

  ----------------------------------------------------------------------------------------------------------------------------------------
  Loss                         Formula
  ---------------------------- -----------------------------------------------------------------------------------------------------------
  **Zero/one**                 $L_{0/1}(y,t) = \begin{cases} 0 & \text{se } y=t \\ 1 & \text{se } y \neq t \end{cases}$

  **Asimmetrica**              $\text{Bal}(y,t) = \begin{cases} \alpha & \text{se } y=1, t=0 \\ \beta & \text{se } y=0, t=1 \end{cases}$

  **Quadratica**               $L_{\text{sq}}(y,t) = (t - y)^2$

  **Assoluta**                 $L_{\text{abs}}(y,t) = \lvert t - y \rvert$
  ----------------------------------------------------------------------------------------------------------------------------------------

> La scelta della loss dipende dal costo dell'errore (es. falsi positivi
> vs falsi negativi in diagnosi medica).

------------------------------------------------------------------------

## 🔍 Separabilità Lineare

-   Se le classi possono essere separate da una retta/iperpiano →
    problema **linearmente separabile**
-   Non sempre è possibile separare perfettamente le classi

------------------------------------------------------------------------

## 📊 Metriche di Valutazione

Per valutare un classificatore (es. "cane vs non-cane"):

``` mermaid
flowchart LR
    subgraph Predetto["Predetto"]
        P1[Positivo]
        P2[Negativo]
    end
    subgraph Reale["Reale"]
        R1[Positivo]
        R2[Negativo]
    end
    R1 -->|TP| P1
    R1 -->|FN| P2
    R2 -->|FP| P1
    R2 -->|TN| P2
```

-   **TP** = True Positive (vero positivo)
-   **FP** = False Positive (falso positivo)
-   **FN** = False Negative (falso negativo)
-   **TN** = True Negative (vero negativo)

------------------------------------------------------------------------

## 🧮 Naive Bayes

### Idea

Classificatore **generativo** per feature discrete
$x \in \{1, \dots, K\}^D$:

-   Assunzione: le feature sono **condizionalmente indipendenti** data
    la classe
-   La densità condizionale di classe diventa un prodotto di densità 1D:

$$p(x \mid y = c, \theta) = \prod_{j=1}^{D} p(x_j \mid y = c, \theta_{jc})$$

### Scelta della Distribuzione per Feature

  Tipo di feature   Distribuzione
  ----------------- -----------------------------------------------------------
  Reale             Gaussiana $\mathcal{N}(x_j \mid \mu_{jc}, \sigma_{jc}^2)$
  Binaria           Bernoulli $\text{Ber}(x_j \mid \mu_{jc})$
  Categorica        Multinoulli $\text{Cat}(x_j \mid \mu_{jc})$

### Classificazione (Regola di Bayes)

$$p(y = c \mid x, \theta) = \frac{p(y = c \mid \theta) \, p(x \mid y = c, \theta)}{\sum_{c'} p(y = c' \mid \theta) \, p(x \mid y = c', \theta)}$$

### Il Trucco Log-Sum-Exp

-   $p(x \mid y = c)$ è spesso un numero **molto piccolo** → underflow
    numerico
-   Soluzione: lavorare in **logaritmo**:

$$\log p(y = c \mid x) = b_c - \log \sum_{c'} e^{b_{c'}}, \qquad b_c = \log p(x \mid y = c) + \log p(y = c)$$

$$\log \sum_c e^{b_c} = B + \log \sum_c e^{b_c - B}, \qquad B = \max_c b_c$$

------------------------------------------------------------------------

## 📍 K-Nearest Neighbors (KNN)

Classificatore **non-parametrico** semplice:

1.  Guarda i **K punti** del training set più vicini al test input $x$
2.  Conta quanti membri di ciascuna classe sono in questo insieme
3.  Restituisce la frazione empirica come stima

$$p(y = c \mid x, \mathcal{D}, K) = \frac{1}{K} \sum_{i \in N_K(x, \mathcal{D})} \mathbb{1}(y_i = c)$$

dove $N_K(x, \mathcal{D})$ sono gli indici dei K punti più vicini a $x$
in $\mathcal{D}$.

``` mermaid
flowchart LR
    A[Test input x] --> B[Trova K vicini più prossimi]
    B --> C[Conta membri per classe]
    C --> D[Frazione empirica = probabilità]
    D --> E[Assegna classe più probabile]
```

### Geometria dei Confini: Celle di Voronoi

Il classificatore k-NN **partiziona lo spazio di input in celle di
Voronoi** (in 2D e 3D): - Ogni punto di training definisce una regione
dello spazio in cui è il punto più vicino - I confini di decisione sono
determinati dalle celle di Voronoi dei punti di training - Per 1-NN,
ogni punto di test è assegnato alla classe del punto di training nella
cui cella di Voronoi cade

### Attributi Simbolici: Distanza di Hamming

Per dati **non metrici** (attributi simbolici/nominali), si usa la
**distanza di Hamming**: - Conta il numero di posizioni in cui due
vettori di attributi differiscono - Non richiede che gli attributi siano
ordinati o numerici

### Complessità e Soluzioni

Il costo al **test time** del k-NN è $O(kdN)$ (dove $d$ è la
dimensionalità e $N$ il numero di esempi di training), che può essere
proibitivo per dataset grandi.

**Rimedi:** - **kd-trees:** struttura dati che organizza i punti in uno
spazio $k$-dimensionale per accelerare la ricerca dei vicini più
prossimi - **Locality Sensitive Hashing (LSH):** tecnica di hashing che
mappa punti simili in bucket simili, permettendo una ricerca
approssimata dei vicini - **Condensing dei dati:** ridurre il training
set rimuovendo i punti che non contribuiscono ai confini di decisione
(es. punti interni ai cluster)

``` mermaid
flowchart TB
    A[Problema: costo O(kdN) al test time] --> B[kd-trees<br/>ricerca accelerata]
    A --> C[LSH<br/>ricerca approssimata]
    A --> D[Condensing<br/>riduzione del training set]
```

------------------------------------------------------------------------

## 📈 Regressione Logistica

La regressione lineare può essere generalizzata alla classificazione
facendo **due cambiamenti**:

1.  Sostituiamo la distribuzione Gaussiana per $y$ con una distribuzione
    **Bernoulli** --- appropriata per la classificazione binaria,
    $y \in \{0, 1\}$:

$$p(y \mid x, w) = \text{Ber}(y \mid \mu(x))$$

dove $\mu(\mathbf{x}) = E[y \mid \mathbf{x}] = p(y = 1 \mid \mathbf{x})$

2.  Calcoliamo una combinazione lineare degli input, come prima, ma la
    passiamo attraverso una funzione che garantisce $0 \le \mu(x) \le 1$
    definendo:

$$\mu(\mathbf{x}) = \text{sigm}(\mathbf{w}^T \mathbf{x})$$

dove $\text{sigm}(\eta)$ si riferisce alla funzione **sigmoid** (o
logistica):

$$\text{sigm}(\eta) = \frac{1}{1 + \exp(-\eta)} = \frac{e^{\eta}}{e^{\eta} + 1}$$

Mettendo insieme questi due passi otteniamo:

$$p(y \mid x, w) = \text{Ber}(y \mid \text{sigm}(w^T x))$$

``` mermaid
flowchart LR
    A[Combinazione lineare w^T x] --> B[Sigmoid sigm(η)]
    B --> C[Probabilità 0 ≤ μ ≤ 1]
    C --> D[Classificazione binaria]
```

------------------------------------------------------------------------

## ⚖️ Bias-Variance Tradeoff

Il **bias-variance tradeoff** è un concetto fondamentale che spiega
perché modelli troppo semplici o troppo complessi performano male.

-   **Bias:** errore dovuto alle assunzioni sbagliate del modello (es.
    assumere linearità quando la relazione è non lineare)
-   **Variance:** errore dovuto alla sensibilità del modello alle
    piccole fluttuazioni nel training set

L'errore di generalizzazione atteso può essere decomposto come:

$$E[\text{errore}] = \text{Bias}^2 + \text{Variance} + \text{Rumore}$$

``` mermaid
flowchart LR
    A[Complessità del modello] --> B{Bias vs Variance}
    B -->|Bassa complessità| C[Alto bias<br/>bassa varianza]
    B -->|Alta complessità| D[Basso bias<br/>alta varianza]
    C --> E[Underfitting]
    D --> F[Overfitting]
```

  ------------------------------------------------------------------------------
                    Bias                   Variance
  ----------------- ---------------------- -------------------------------------
  **Definizione**   Errore da assunzioni   Errore da sensibilità ai dati
                    sbagliate              

  **Modello         Alto                   Basso
  semplice**                               

  **Modello         Basso                  Alto
  complesso**                              

  **Riduzione**     Aumentare complessità  Regolarizzazione, ensemble, più dati
  ------------------------------------------------------------------------------

> La **regolarizzazione** e i **metodi ensemble** (bagging, random
> forests) sono tecniche per ridurre la varianza.

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 03_valutazione_modelli.md ===== -->
```

------------------------------------------------------------------------

# PARTE II --- Valutazione, metriche e strumenti

# 03 --- Valutazione dei Modelli

> *Riferimento: Witten & Frank, Data Mining, Ch. 5*

------------------------------------------------------------------------

## 🎯 Perché Valutare?

-   L'**errore sui dati di training** non è un buon indicatore della
    performance su dati futuri
-   Serve un modo per **predire i limiti di performance** basato su
    esperimenti con dati indipendenti (test set)

``` mermaid
flowchart LR
    A[Errore su training] -->|Non affidabile| B[Errore su test set]
    B --> C[Stima della generalizzazione]
```

### Training e Testing

-   **Error rate** = proporzione di errori sull'intero insieme di
    istanze
-   **Resubstitution error** = error rate ottenuto sui dati di training
    -   È **ottimistico** (il modello ha già "visto" i dati)
    -   Ma è sempre utile conoscerlo

------------------------------------------------------------------------

## ✂️ Holdout Estimation

Quando i dati sono limitati:

-   Si riserva una parte per il test e si usa il resto per il training
-   **Di solito:** un terzo per il test, il resto per il training

**Problema:** i campioni potrebbero non essere rappresentativi (es. una
classe manca nel test o nel training)

**Soluzione avanzata:** la **stratificazione** garantisce che ogni
classe sia rappresentata con proporzioni approssimativamente uguali in
entrambi i sottoinsiemi.

------------------------------------------------------------------------

## 🔁 Repeated Holdout

-   La stima holdout può essere resa più affidabile **ripetendo** il
    processo con diversi sottoinsiemi
-   In ogni iterazione si seleziona casualmente una proporzione per il
    training (possibilmente con stratificazione)
-   Gli error rate delle diverse iterazioni vengono **mediati**

> **Limite:** i diversi test set si sovrappongono. Possiamo evitarlo?

------------------------------------------------------------------------

## 🔄 Cross-Validation

La cross-validation **evita la sovrapposizione** dei test set:

1.  **Split** dei dati in $k$ sottoinsiemi di uguale dimensione
2.  Ogni sottoinsieme viene usato a turno per il test, il resto per il
    training

``` mermaid
flowchart TB
    subgraph Fold1["Fold 1"]
        T1[Test]
        R1[Train]
    end
    subgraph Fold2["Fold 2"]
        R2[Train]
        T2[Test]
    end
    subgraph Fold3["Fold 3"]
        R3[Train]
        T3[Test]
    end
    subgraph FoldK["Fold K"]
        R4[Train]
        T4[Test]
    end
```

-   Chiamata **k-fold cross-validation**
-   Spesso i sottoinsiemi sono **stratificati** prima della CV
-   Le stime di errore vengono **mediate**

### Standard: Stratified 10-Fold CV

-   Metodo standard di valutazione: **stratified ten-fold
    cross-validation**
-   Esperimenti estesi mostrano che è la scelta migliore per una stima
    accurata
-   La stratificazione **riduce la varianza** della stima
-   Ancora meglio: **repeated stratified CV** (es. 10-fold ripetuta 10
    volte e mediata)

------------------------------------------------------------------------

## 🎯 Leave-One-Out CV

Forma particolare di cross-validation:

-   Il numero di fold è uguale al numero di istanze di training
-   Per $n$ istanze di training, si costruisce il classificatore $n$
    volte
-   I risultati di tutti gli $n$ giudizi vengono mediati

**Vantaggi:** - Fa il miglior uso dei dati per il training - Non
coinvolge subsampling casuale - Non ha senso ripeterla (stesso risultato
ogni volta)

**Svantaggi:** - Molto costosa computazionalmente - La stratificazione
**non è possibile** (c'è solo un'istanza nel test set!)

> **Esempio estremo:** dataset casuale diviso in due classi uguali. Il
> vero error rate è 50%. Ma in ogni fold di leave-one-out la classe
> opposta all'istanza di test è in maggioranza → la stima
> Leave-One-Out-CV è **100% di errore**!

------------------------------------------------------------------------

## 🎲 Il Bootstrap

-   La cross-validation usa **sampling senza sostituzione**
-   Il bootstrap usa **sampling con sostituzione** per formare il
    training set

``` mermaid
flowchart LR
    A[Dataset originale n istanze] --> B[Sampling con sostituzione]
    B --> C[Training set n istanze]
    B --> D[Test set: istanze non selezionate]
```

### Il Bootstrap 0.632

-   Un'istanza ha probabilità $1 - 1/n$ di non essere scelta
-   Quindi la probabilità di finire nel test data è:

$$\left(1 - \frac{1}{n}\right)^n \approx 0.368$$

-   Il test set conterrà circa il **36.8%** delle istanze
-   Il training set conterrà circa il **63.2%** delle istanze

### Stima dell'Errore

L'errore sul test data è **pessimistico** (training su solo \~63% delle
istanze). Si combina con l'errore di resubstitution:

$$err = 0.632 \cdot e_{\text{test}} + 0.368 \cdot e_{\text{training}}$$

-   L'errore di resubstitution ha **meno peso** dell'errore sul test
    data
-   Si ripete la procedura più volte e si mediano i risultati

### Problemi del Bootstrap

> **Esempio:** dataset casuale diviso in due classi uguali. Il vero
> error rate è 50%. Un perfetto memorizzatore ottiene 0% di
> resubstitution error e \~50% di errore sul test. Stima bootstrap:
> $err = 0.632 \cdot 50\% + 0.368 \cdot 0\% = 31.6\%$ → **ottimistica**
> (il vero errore atteso è 50%).

-   Probabilmente il **miglior modo** di stimare la performance per
    dataset **molto piccoli**

------------------------------------------------------------------------

## ⚙️ Selezione degli Iperparametri

-   **Iperparametro** = parametro che può essere regolato per
    ottimizzare la performance di un algoritmo di apprendimento
-   Diverso dal parametro base che fa parte del modello (es.
    coefficiente in regressione lineare)
-   **Esempio di iperparametro:** $k$ nel classificatore k-nearest
    neighbor

> ⚠️ **Non è permesso** guardare i dati di test finali per scegliere il
> valore di questo parametro! Regolare l'iperparametro sui dati di test
> porta a stime di performance **ottimistiche**.

### Iperparametri e Cross-Validation

-   La k-fold CV esegue $k$ diverse valutazioni train-test
-   Il processo di tuning con validation set deve essere applicato
    **separatamente** a ciascuno dei $k$ training set
-   Possono essere selezionati $k$ valori diversi di iperparametro
-   Questo è OK: il tuning degli iperparametri fa parte del **processo
    di apprendimento**
-   La cross-validation valuta la qualità del **processo di
    apprendimento**, non di un particolare modello

``` mermaid
flowchart TB
    A[Training set] --> B[Validation set per tuning]
    A --> C[Test set finale]
    B --> D[Seleziona iperparametro]
    D --> E[Addestra modello finale]
    E --> F[Valuta su test set]
```

------------------------------------------------------------------------

## 🔧 Nota sul Parameter Tuning

Alcuni schemi di apprendimento operano in **due stadi**: - **Stadio 1:**
costruire il modello base - **Stadio 2:** ottimizzare le impostazioni
dei parametri

> ⚠️ I dati di test **non possono essere usati** per il parameter
> tuning!

**Procedura corretta** usa tre insiemi scelti indipendentemente: -
**Training data** --- per costruire il modello - **Validation data** ---
per ottimizzare i parametri - **Test data** --- per la valutazione
finale

------------------------------------------------------------------------

## 🪆 Nested Cross-Validation

Cosa fare quando i training set sono **molto piccoli**, così che le
stime di performance su un validation set sono inaffidabili?

Possiamo usare la **nested cross-validation** (costosa!): - Per ogni
training set della cross-validation "esterna" (outer): - Esegui
cross-validazioni "interne" (inner) per scegliere il miglior valore
dell'iperparametro

``` mermaid
flowchart TB
    subgraph Outer["Outer CV (k-fold)"]
        direction TB
        O1[Training set 1]
        O2[Training set 2]
        O3[Training set k]
    end
    subgraph Inner["Inner CV (p-fold)"]
        direction TB
        I1[Seleziona iperparametro]
        I2[Seleziona iperparametro]
        I3[Seleziona iperparametro]
    end
    O1 --> I1
    O2 --> I2
    O3 --> I3
```

-   La **outer cross-validation** è usata per stimare la qualità del
    processo di apprendimento
-   Le **inner cross-validations** sono usate per scegliere i valori
    degli iperparametri
-   Le inner cross-validations fanno parte del **processo di
    apprendimento**!

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 04_test_statistici.md ===== -->
```
# 04 --- Intervalli di Confidenza e Test Statistici

> *Riferimento: Witten & Frank, Data Mining, Ch. 5*

------------------------------------------------------------------------

## 📊 Intervalli di Confidenza

Possiamo dire che $p$ giace entro un certo intervallo specificato con
una certa confidenza.

**Esempio:** $S = 750$ successi in $N = 1000$ prove - Success rate
stimato: 75% - Quanto è vicino al vero success rate $p$? - **Risposta:**
con confidenza 80%, $p \in [73.2, 76.7]$

**Altro esempio:** $S = 75$ e $N = 100$ - Success rate stimato: 75% -
Con confidenza 80%, $p \in [69.1, 80.1]$

------------------------------------------------------------------------

## 📐 Media e Varianza

Per una singola prova di Bernoulli con success rate $p$: - **Media:**
$p$ - **Varianza:** $p(1-p)$

Per $N$ prove, il success rate atteso è $f = S/N$: - $f$ è una variabile
casuale con media $p$ e varianza $p(1-p)/N$ - Per $N$ abbastanza grande,
la distribuzione di $f$ segue una **distribuzione normale**

La probabilità che una variabile casuale $X$ con media 0 giaccia entro
un intervallo di confidenza $[-z \le X \le z]$ di larghezza $2z$ è:

$$\Pr[-z \le X \le z] = 1 - 2 \times \Pr[X \ge z]$$

------------------------------------------------------------------------

## 📋 Limiti di Confidenza

Limiti di confidenza per la distribuzione normale con media 0 e varianza
1: $\Pr[X \ge z]$

  $\Pr[X \ge z]$   $z$
  ---------------- ------
  0.1%             3.09
  0.5%             2.58
  1%               2.33
  5%               1.65
  10%              1.28
  20%              0.84
  40%              0.25

Quindi: $\Pr[-1.65 \le X \le 1.65] = 90\%$

> Per usare questo approccio con la variabile casuale $f$, dobbiamo
> **ridurre** $f$ a media 0 e varianza unitaria. Confidenza desiderata
> 90% → valore per accedere alla tabella dato da $(1-90\%)/2$.

------------------------------------------------------------------------

## 🔄 Trasformazione di $f$

Valore trasformato per $f$:

$$\frac{f - p}{\sqrt{p(1-p)/N}}$$

(sottrai la media e dividi per la deviazione standard)

Equazione risultante:

$$\Pr\left[-z \le \frac{f - p}{\sqrt{p(1-p)/N}} \le z\right] = c$$

Risolvendo per $p$ si ottengono i limiti superiore e inferiore
dell'intervallo di confidenza:

$$p = \left(f + \frac{z^2}{2N} \pm z \sqrt{\frac{f}{N} - \frac{f^2}{N} + \frac{z^2}{4N^2}}\right) / \left(1 + \frac{z^2}{N}\right)$$

### Esempi

    · f = 75%, N = 1000, c = 80% (z = 1.28): p ∈ [0.732, 0.767]
    · f = 75%, N = 100,  c = 80% (z = 1.28): p ∈ [0.691, 0.801]

> ⚠️ L'assunzione di distribuzione normale è valida solo per $N$ grande
> ($N > 100$). Per $N = 10$: $p \in [0.549, 0.881]$ (da prendere con
> cautela).

------------------------------------------------------------------------

## ⚖️ Confronto tra Schemi di Apprendimento

-   Domanda frequente: **quale dei due schemi di apprendimento performa
    meglio?**
-   Nota: questo è **dipendente dal dominio**!
-   Modo ovvio: confrontare gli error rate calcolati con stime 10-fold
    CV
-   **Problema:** varianza nella stima su una singola 10-fold CV
-   La varianza può essere ridotta usando **repeated CV**
-   Tuttavia, non sappiamo ancora se i risultati sono affidabili

------------------------------------------------------------------------

## 🧪 Test di Significatività

-   I test di significatività ci dicono quanto possiamo essere
    **fiduciosi** che ci sia davvero una differenza tra i due schemi
-   **Ipotesi nulla:** non c'è differenza "reale"
-   **Ipotesi alternativa:** c'è una differenza
-   Un test di significatività misura quanta evidenza c'è a favore del
    **rifiuto dell'ipotesi nulla**

``` mermaid
flowchart LR
    A[Due schemi di apprendimento] --> B[10-fold CV per ciascuno]
    B --> C[Due medie delle stime]
    C --> D{Le medie differiscono significativamente?}
    D -->|Sì| E[Rifiuta ipotesi nulla]
    D -->|No| F[Non rifiuta ipotesi nulla]
```

------------------------------------------------------------------------

## 📈 Paired t-test

Il **t-test di Student** dice se le medie di due campioni sono
significativamente diverse.

**Assunzione teorica iniziale:** - La fornitura di dati è illimitata -
Si prendono diversi dataset della stessa dimensione - Si ottiene una
stima di accuratezza per ogni dataset usando cross-validation (es.
10-fold CV) - Ogni esperimento di CV produce una stima di errore
**diversa e indipendente**

### Distribuzione delle Medie

-   $x_1, x_2, \dots, x_k$ e $y_1, y_2, \dots, y_k$ sono i $2k$ campioni
    per i $k$ dataset diversi, usando i due schemi
-   $m_x$ e $m_y$ sono le medie
-   Con abbastanza campioni, la media di un insieme di campioni
    indipendenti è **normalmente distribuita**
-   La varianza è sconosciuta:

$$\sigma^2 = \frac{1}{k} \sum_{i=1}^{k} (x_i - m_x)^2$$

-   Le varianze stimate delle medie sono $\sigma_x^2/k$ e $\sigma_y^2/k$
-   Se $\mu_x$ e $\mu_y$ sono le medie vere (sconosciute):

$$\frac{m_x - \mu_x}{\sqrt{\sigma_x^2/k}} \qquad \frac{m_y - \mu_y}{\sqrt{\sigma_y^2/k}}$$

sono approssimativamente normalmente distribuite con media 0, varianza
1.

### Distribuzione di Student

Con campioni piccoli ($k < 100$) la media segue la **distribuzione di
Student** con $k - 1$ gradi di libertà.

**9 gradi di libertà** per $k = 10$ stime (fold):

  $\Pr[X \ge z]$   $z$
  ---------------- ------
  0.1%             4.30
  0.5%             3.25
  1%               2.82
  5%               1.83
  10%              1.38
  20%              0.88

### Distribuzione delle Differenze

-   Sia $m_d = m_x - m_y$
-   La differenza delle medie ($m_d$) ha anch'essa una distribuzione di
    Student con $k - 1$ gradi di libertà
-   Sia $\sigma_d^2$ la varianza della differenza
-   La versione standardizzata di $m_d$ è chiamata **t-statistica**:

$$t = \frac{m_d}{\sqrt{\sigma_d^2/k}}$$

Usiamo $t$ per eseguire il t-test.

### Eseguire il Test

-   Dobbiamo verificare se $m_d$ è significativamente diverso da 0
-   Per un dato livello di confidenza, verifichiamo se la differenza
    attuale supera il limite di confidenza
-   Fissiamo un livello di confidenza $c\%$ (generalmente 5% o 1%)
-   Se una differenza è significativa al livello $c\%$, c'è una
    probabilità $(100-c)\%$ che ci sia davvero una differenza

------------------------------------------------------------------------

## 🔓 Osservazioni Non Appaiate

-   Se le stime CV provengono da **dataset diversi**, non sono più
    appaiate
-   (es. abbiamo usato $k$ stime per uno schema e $j$ stime per l'altro)
-   Allora serve un **t-test non appaiato** con $\min(k, j) - 1$ gradi
    di libertà
-   La t-statistica diventa:

$$t = \frac{m_x - m_y}{\sqrt{\frac{\sigma_x^2}{k} + \frac{\sigma_y^2}{j}}}$$

------------------------------------------------------------------------

## ⚠️ t-test in Stime Reali: Stime Dipendenti

**Caso reale:** un singolo dataset disponibile di dimensione limitata: -
Serve riusare i dati - Es. eseguire repeated n-fold CV (diverse
randomizzazioni) sullo stesso dataset - I campioni diventano
**dipendenti** → differenze insignificanti possono diventare
significative - Il t-test standard **non può più essere applicato**

### Corrected Resampled t-test

Viene applicato un test euristico chiamato **corrected resampled
t-test**:

-   Assunzione: si usa il metodo **repeated hold-out** (invece di CV)
-   $k$ ripetizioni di diversi split casuali dello stesso dataset. Per
    ogni ripetizione:
    -   $n_1$ istanze per il training, $n_2$ per il test
    -   Differenze $d_i$ calcolate dalla performance sui dati di test

La stessa statistica modificata può essere usata con **repeated CV**
(caso speciale di repeated holdout dove i test set individuali per una
CV non si sovrappongono).

**Per:** 10-fold cross-validation ripetuta 10 volte: - $k = 100$ -
$n_2/n_1 = 0.1/0.9$ - $\sigma_d^2$ basata su 100 differenze

### La Statistica del Corrected Resampled t-test

La statistica del corrected resampled t-test tiene conto della
**dipendenza** tra le stime:

$$t = \frac{m_d}{\sqrt{\left(\frac{1}{k} + \frac{n_2}{n_1}\right) \sigma_d^2}}$$

dove: - $m_d$ è la media delle differenze - $\sigma_d^2$ è la varianza
delle differenze - $n_1$ è il numero di istanze di training - $n_2$ è il
numero di istanze di test - Il termine $\frac{n_2}{n_1}$ corregge la
sovrastima della varianza dovuta alla dipendenza

``` mermaid
flowchart TB
    A[Dataset limitato] --> B[Repeated CV / holdout]
    B --> C[Stime dipendenti]
    C --> D[Standard t-test non valido]
    D --> E[Corrected resampled t-test]
    E --> F[Statistica modificata]
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 05_probabilita_e_costi.md ===== -->
```
# 05 --- Predizione di Probabilità e Costi

> *Riferimento: Witten & Frank, Data Mining, Ch. 5*

------------------------------------------------------------------------

## 🎯 Predire le Probabilità

-   Misura di performance finora: **success rate** (chiamata anche **0-1
    loss**)
-   La maggior parte dei classificatori produce **probabilità di
    classe**
-   Potremmo voler verificare l'accuratezza delle stime di probabilità
-   La 0-1 loss **non è la cosa giusta** da usare in questi casi

```{=html}
<!-- -->
```
    Caso di loss 0: se la predizione è corretta
    Caso di loss 1: se la predizione è incorretta

------------------------------------------------------------------------

## 📉 Quadratic Loss Function

-   $p_1, \dots, p_k$ sono le stime di probabilità per un'istanza
    rispetto a ciascuna classe $c_1, \dots, c_k$
-   $c$ è l'indice della classe effettiva dell'istanza
-   Il risultato effettivo per un'istanza è rappresentato come un
    vettore $a_1, \dots, a_k = 0$, tranne $a_c$ che è 1
-   La **loss quadratica** per una singola classificazione è:

$$\sum_{j=1}^{k} (p_j - a_j)^2 = \underbrace{\sum_{j \neq c} p_j^2}_{\text{contributo predizioni incorrette}} + \underbrace{(1 - p_c)^2}_{\text{contributo predizione corretta}}$$

> Per un test set con più istanze, la loss viene sommata su tutte.

------------------------------------------------------------------------

## 📊 Informational Loss Function

Un altro criterio per la valutazione della predizione probabilistica è
la **loss informazionale**: $-\log_2(p_c)$, dove $c$ è l'indice della
classe effettiva dell'istanza.

-   Numero di bit richiesti per comunicare la classe effettiva $c$
-   Siano $p_1^*, \dots, p_k^*$ le vere probabilità di classe
-   Il valore atteso per la loss informazionale è:

$$\sum_{j=1}^{k} p_j^* \log_2 p_j$$

-   **Giustificazione:** minimizzata quando $p_j = p_j^*$
-   **Difficoltà:** problema della *frequenza zero* --- la probabilità
    assegnata a un evento non è mai zero per evitare infinito come
    risultato del logaritmo
    -   Viene usato lo **stimatore di Laplace**

------------------------------------------------------------------------

## 💰 Contare il Costo

-   Le valutazioni discusse finora **non tengono conto del costo** di
    prendere decisioni sbagliate
-   Diversi tipi di errori di classificazione possono comportare **costi
    diversi**
-   Ottimizzare il tasso di classificazione senza considerare il costo
    degli errori può portare a **risultati strani** (es. dataset
    sbilanciati)
-   La valutazione per accuratezza di classificazione assume tacitamente
    **costo degli errori uguale**

**Esempi:** - Profiling terroristi: "Non terrorista" corretto il 99.99%
delle volte - Decisioni di prestito, diagnosi - Mailing promozionale
(perdita di business)

### La Matrice di Confusione

La **matrice di confusione** (classificazione binaria):

                        Predetto: Yes (classe rara)   Predetto: No
  ------------------ -- ----------------------------- ---------------------
  **Attuale: Yes**      True Positive (TP)            False Negative (FN)
  **Attuale: No**       False Positive (FP)           True Negative (TN)

-   I buoni risultati corrispondono a **grandi numeri lungo la diagonale
    principale** e piccoli (idealmente zero) elementi fuori diagonale
-   Diversi costi di misclassificazione possono essere assegnati a false
    positives e false negatives
-   Ci sono molti altri tipi di costo! Es. il costo di raccogliere i
    dati di training

------------------------------------------------------------------------

## 📋 La Statistica Kappa

-   Due matrici di confusione per un problema a 3 classi: predittore
    effettivo (sinistra) vs predittore casuale (destra)
-   Numero di successi: somma delle voci sulla diagonale ($D$)
-   **Statistica Kappa:**

$$\kappa = \frac{\text{success rate del predittore effettivo} - \text{success rate del predittore casuale}}{1 - \text{success rate del predittore casuale}}$$

-   Misura il **miglioramento relativo** rispetto al predittore casuale:
    -   **1** = accuratezza perfetta
    -   **0** = non facciamo meglio del caso

------------------------------------------------------------------------

## 📐 Contare il Costo in Percentuali

Dato il totale degli esempi positivi effettivi: - **True Positive Rate
(TPR):** $TPR = TP/(TP+FN)$ - **False Negative Rate (FNR):**
$FNR = FN/(TP+FN)$

Dato il totale degli esempi negativi effettivi: - **True Negative Rate
(TNR):** $TNR = TN/(TN+FP)$ - **False Positive Rate (FPR):**
$FPR = FP/(TN+FP)$

Dato il numero totale di classificazioni: - **Overall Success Rate:**
$SuccR = (TP+TN)/(TP+TN+FP+FN)$ - **Error Rate:** $1 - SuccR$

------------------------------------------------------------------------

## 🎯 Precision, Recall, F-Measure

Precision e Recall sono due metriche ampiamente usate quando la
**rilevazione di successo di una classe** è più significativa della
rilevazione delle altre. Largamente usate in **Information Retrieval
(IR)**.

### Precision

$$precision = \frac{TP}{TP+FP}$$

-   Frazione di esempi che risultano effettivamente positivi tra gli
    esempi classificati come positivi
-   Più alta è la precision, più basso è il FPR
-   IR: percentuale di documenti recuperati che sono rilevanti

### Recall

$$recall = \frac{TP}{TP+FN} \equiv \text{True Positive Rate (TPR)}$$

-   Frazione di esempi positivi correttamente predetti
-   Recall grande → pochissimi esempi positivi classificati male
-   Costruire un modello che massimizzi **sia** precision che recall è
    la sfida chiave degli algoritmi di classificazione

### F-Measure

$$F_1 = \frac{2 \times recall \times precision}{recall + precision}$$

-   Riassume Precision e Recall
-   Un valore alto di $F_1$ garantisce che sia precision che recall sono
    presumibilmente alti

$$F_\beta = \frac{(1 + \beta^2) \times recall \times precision}{recall + \beta^2 \times precision}$$

-   Generalizza la $F_1$-measure

``` mermaid
flowchart LR
    A[Predizioni] --> B[Precision<br/>TP/(TP+FP)]
    A --> C[Recall<br/>TP/(TP+FN)]
    B --> D[F1-measure]
    C --> D
```

------------------------------------------------------------------------

## 💼 Classificazione con Costi

Se i costi sono noti → possono essere rappresentati nella **matrice di
confusione**.

Due matrici di costo (due/tre classi): - I costi di misclassificazione
sostituiscono il numero di istanze - Nella valutazione cost-sensitive
dei metodi di classificazione: - Il success rate è sostituito dal
**costo medio per predizione** - Il costo è dato dalla voce appropriata
nella matrice di costo - I costi sono **ignorati** quando si fanno
predizioni, ma **considerati** quando le si valuta

------------------------------------------------------------------------

## 🎲 Classificazione Cost-Sensitive

### Il caso delle predizioni probabilistiche

-   Si possono considerare i costi quando si fanno predizioni
-   **Idea di base:** predire la classe ad alto costo solo quando si è
    molto fiduciosi sulla predizione
-   Dati: probabilità di classe predette
-   Normalmente prediciamo la classe più probabile
-   Qui dovremmo fare la predizione che **minimizza il costo atteso**
-   **Costo atteso:** prodotto scalare del vettore delle probabilità di
    classe e della colonna appropriata nella matrice di costo
-   Scegli la colonna (classe) che minimizza il costo atteso
-   Questo è l'approccio **minimum-expected cost** alla classificazione
    cost-sensitive

------------------------------------------------------------------------

## 🛠️ Apprendimento Cost-Sensitive

-   Finora non abbiamo considerato i costi al **tempo di training**
-   La maggior parte degli schemi di apprendimento **non esegue**
    apprendimento cost-sensitive
-   Generano lo stesso classificatore indipendentemente dai costi
    assegnati alle diverse classi
-   **Esempio:** standard decision tree learner

**Metodi semplici per l'apprendimento cost-sensitive:** - **Resampling**
delle istanze nel training set (non nel test set) secondo i costi -
Aumentare il numero di istanze per cui faremmo meno errori -
**Weighting** delle istanze secondo i costi (es. falso positivo/negativo
per il caso binario) - I costi sono ignorati al tempo di predizione

------------------------------------------------------------------------

## 📈 Lift Charts

In pratica, i costi sono **raramente noti**. Le decisioni sono
solitamente prese confrontando possibili scenari.

**Esempio:** mailing promozionale a 1,000,000 di famiglie - **Scenario
1:** mail a tutti; 0.1% risponde (1000) - **Scenario 2:** il tool di
data mining identifica un sottoinsieme di 100,000 più promettenti, 0.4%
di questi risponde (400) - 40% delle risposte per il 10% del costo può
ripagare - **Scenario 3:** identifica un sottoinsieme di 400,000 più
promettenti, 0.2% risponde (800)

L'aumento del tasso di risposta (un fattore di quattro nell'esempio) è
noto come **lift factor** (prodotto dal tool di apprendimento).

Dato un metodo di apprendimento probabilistico che produce probabilità
per la classe predetta di ogni membro del test set: - **Obiettivo:**
trovare sottoinsiemi di istanze di test con una proporzione di istanze
positive più alta che nell'intero test - Un **lift chart** permette un
confronto visivo

### Generare un Lift Chart

1.  Ordina le istanze in base alla probabilità predetta di essere
    positive
2.  Per trovare un campione di una data dimensione e la proporzione di
    istanze positive, leggi il numero richiesto di istanze dalla lista,
    partendo dall'alto
3.  Calcola il **lift factor**:
    -   Calcola la proporzione di successo del campione (numero di
        istanze positive nel campione diviso per la dimensione del
        campione)
    -   Dividi per la proporzione di successo per l'intero test set

> **Esempio:** prendendo le prime 10 istanze: 80% / 33% = **2.4 lift
> factor**

Ripetendo il calcolo del lift factor per campioni di dimensioni diverse
si può tracciare un lift chart: - **x axis:** dimensione del campione -
**y axis:** numero di true positives

------------------------------------------------------------------------

## 📉 Curve ROC

-   Le curve ROC sono simili ai lift charts
-   Tecniche grafiche usate quando il learner cerca di selezionare
    campioni di istanze di test con un'alta proporzione di positivi
-   Sta per **"receiver operating characteristic"**
-   Usate nella rilevazione di segnali per mostrare il tradeoff tra hit
    rate e false alarm rate su un canale rumoroso

**Differenze rispetto al lift chart:** - **y axis:** percentuale di true
positives nel campione (non numero assoluto) - **x axis:** percentuale
di false positives nel campione (non dimensione del campione)

``` mermaid
flowchart LR
    A[Curva ROC] --> B[y: % True Positives]
    A --> C[x: % False Positives]
    B --> D[Tradeoff hit rate vs false alarm]
    C --> D
```

### Cross-Validation e Curve ROC

Metodo semplice per ottenere una curva ROC usando la
cross-validation: 1. Raccogli le probabilità per le istanze nei test
fold (10 test fold in una 10-fold CV) insieme alle vere etichette di
classe 2. Ordina le istanze secondo le probabilità in un'unica lista di
ranking 3. Questo metodo è implementato in **WEKA**

> È solo una possibilità --- la più semplice e più usata. Un'altra
> possibilità (più costosa) è generare una curva ROC per ogni fold e
> mediarle.

### La Curva AUC

Per riassumere le curve ROC in una singola quantità si usa l'**Area
under the ROC curve (AUC)**.

-   In generale, più grande è l'area, migliore è il modello
-   **Interpretazione:** probabilità che il classificatore classifichi
    un'istanza positiva scelta casualmente sopra un'istanza negativa
    scelta casualmente
-   a)  Valore ottenuto da una lista di istanze di test ordinate in
        ordine decrescente di probabilità predetta della classe positiva
-   b)  Per ogni istanza positiva, conta quante negative sono
        classificate sotto di essa (aumenta il conteggio di 1/2 se
        positive e negative sono a pari rango)
-   c)  Calcola il totale di questi conteggi (questa è la statistica U)
-   d)  Dividi U per il prodotto del numero di istanze positive e
        negative nel test set (questa è la statistica ρ)

### Curve ROC per Due Schemi

Idealmente opera sempre in un punto che giace sul **limite superiore
dello scafo convesso**: - Per un campione piccolo e focalizzato, usa il
metodo A - Per uno più grande, usa il metodo B - In mezzo, scegli tra A
e B con probabilità appropriate

### Lo Scafo Convesso

Dati due schemi di apprendimento possiamo raggiungere **qualsiasi punto
sullo scafo convesso**!

-   TP e FP rates per lo schema 1: $t_1$ e $f_1$
-   TP e FP rates per lo schema 2: $t_2$ e $f_2$
-   Se lo schema 1 è usato per predire $100 \times q\%$ dei casi e lo
    schema 2 per il resto:
    -   **TP rate combinato:** $q \times t_1 + (1-q) \times t_2$
    -   **FP rate combinato:** $q \times f_1 + (1-q) \times f_2$

------------------------------------------------------------------------

## 📊 Recall-Precision Curve

La Recall-Precision Curve è simile alla curva ROC e al lift chart,
adottata in **Information Retrieval**.

-   Es. data la tabella usata per il lift chart --- la lista di sì e no
    rappresenta una lista classificata di documenti recuperati e se
    erano rilevanti o no
-   L'intera collezione conteneva un totale di 40 documenti rilevanti
-   "recall at 10" si riferisce al recall per i primi dieci documenti,
    cioè 8/40 = 20%
-   "precision at 10" sarebbe 8/10 = 80%

Gli esperti di IR usano curve recall-precision che tracciano l'una
contro l'altra, per diversi numeri di documenti recuperati, proprio come
le curve ROC e i lift charts --- tranne che poiché gli assi sono
diversi, le curve sono **iperboliche** e il punto operativo desiderato è
verso l'**alto a destra**.

------------------------------------------------------------------------

## 📋 Riepilogo delle Misure

  ---------------------------------------------------------------------------------------------------
                       Dominio              Plot           Spiegazione
  -------------------- -------------------- -------------- ------------------------------------------
  **Lift chart**       Marketing            TP vs          $\frac{TP+FP}{TP+FP+TN+FN} \times 100\%$
                                            dimensione del 
                                            sottoinsieme   

  **ROC curve**        Comunicazioni        TP rate vs FP  $\frac{TP}{TP+FN} \times 100\%$,
                                            rate           $\frac{FP}{FP+TN} \times 100\%$

  **Recall-precision   Information          Recall vs      $\frac{TP}{TP+FN}$, $\frac{TP}{TP+FP}$
  curve**              retrieval            Precision      
  ---------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 06_valutazione_numerica.md ===== -->
```
# 06 --- Valutazione della Predizione Numerica

> *Riferimento: Witten & Frank, Data Mining, Ch. 5*

------------------------------------------------------------------------

## 🎯 Valutare la Predizione Numerica

-   **Stesse strategie:** test set indipendente, cross-validation, test
    di significatività, ecc.
-   **Differenza:** le misure di errore --- gli errori non sono
    semplicemente presenti o assenti; vengono in **diverse dimensioni**
-   Valori target effettivi: $a_1, a_2, \dots, a_n$
-   Valori target predetti: $p_1, p_2, \dots, p_n$
-   **Misura più popolare:** mean-squared error (chiamata anche
    mean-squared loss)
    -   Facile da manipolare matematicamente

$$MSE = \frac{(p_1 - a_1)^2 + \cdots + (p_n - a_n)^2}{n}$$

------------------------------------------------------------------------

## 📐 Altre Misure

### Root Mean-Squared Error (RMSE)

$$RMSE = \sqrt{\frac{(p_1 - a_1)^2 + \cdots + (p_n - a_n)^2}{n}}$$

### Mean Absolute Error (MAE)

$$MAE = \frac{|p_1 - a_1| + \cdots + |p_n - a_n|}{n}$$

> Il **mean absolute error** è meno sensibile agli **outlier** (istanze
> il cui errore di predizione è più grande delle altre) rispetto al
> mean-squared error.

``` mermaid
flowchart LR
    A[Errore di predizione] --> B[MSE<br/>media dei quadrati]
    A --> C[RMSE<br/>radice di MSE]
    A --> D[MAE<br/>media dei valori assoluti]
    B --> E[Sensibile agli outlier]
    D --> F[Meno sensibile agli outlier]
```

------------------------------------------------------------------------

## 📈 Miglioramento sulla Media

A volte i valori di errore **relativi** sono più appropriati.

> **Esempio:** un errore del 10% è ugualmente importante sia che sia un
> errore di 50 quando si predice 500, sia che sia un errore di 0.2
> quando si predice 2.

L'errore è reso relativo a quello che sarebbe stato se fosse stato usato
un **predittore semplice**. Quanto migliora lo schema rispetto a predire
semplicemente il valore medio dai dati di training?

### Relative Squared Error

$$\frac{(p_1 - a_1)^2 + \cdots + (p_n - a_n)^2}{(a_1 - \bar{a})^2 + \cdots + (a_n - \bar{a})^2}$$

### Root Relative Squared Error

$$\sqrt{\frac{(p_1 - a_1)^2 + \cdots + (p_n - a_n)^2}{(a_1 - \bar{a})^2 + \cdots + (a_n - \bar{a})^2}}$$

### Relative Absolute Error

$$\frac{|p_1 - a_1| + \cdots + |p_n - a_n|}{|a_1 - \bar{a}| + \cdots + |a_n - \bar{a}|}$$

> dove $\bar{a}$ sta per il valore medio sui dati di training, cioè il
> valore medio di tutti i valori target (reali).

------------------------------------------------------------------------

## 🔗 Coefficiente di Correlazione

Misura la **correlazione statistica** tra i valori predetti e i valori
effettivi.

$$r = \frac{S_{PA}}{\sqrt{S_P S_A}}$$

dove:

$$S_{PA} = \frac{\sum_i (p_i - \bar{p})(a_i - \bar{a})}{n - 1}, \qquad S_P = \frac{\sum_i (p_i - \bar{p})^2}{n - 1}, \qquad S_A = \frac{\sum_i (a_i - \bar{a})^2}{n - 1}$$

(qui $\bar{a}$ è il valore medio sui dati di test)

-   **Indipendente dalla scala**, tra $-1$ e $+1$
    -   **1** = risultati perfettamente correlati
    -   **0** = nessuna correlazione
    -   **-1** = risultati perfettamente correlati negativamente (la
        relazione tra due variabili è esattamente opposta tutto il
        tempo)
-   **Buona performance porta a valori grandi verso 1!**

------------------------------------------------------------------------

## 🤔 Quale Misura Usare?

-   **Meglio guardarle tutte**
-   Spesso non importa

  Misura                    A       B       C       D
  ------------------------- ------- ------- ------- -------
  Correlation coefficient   0.88    0.88    0.89    0.91
  Relative absolute error   42.2%   57.2%   39.4%   35.8%
  Root rel squared error    43.1%   40.1%   34.8%   30.4%
  Mean absolute error       41.3    38.5    33.4    29.2
  Root mean-squared error   67.8    91.7    63.3    57.4

-   **D** migliore
-   **C** secondo migliore
-   **A, B** discutibili

> **Nota:** quando si confrontano due diversi schemi di apprendimento
> per la predizione numerica, la metodologia presentata si applica
> ancora. Unica differenza: il success rate è sostituito dalla misura di
> performance appropriata (es. root mean-squared error) quando si esegue
> il test di significatività.

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 07_weka.md ===== -->
```
# 07 --- WEKA

> *Waikato Environment for Knowledge Analysis*

------------------------------------------------------------------------

## 🧰 Cos'è WEKA

-   **Collezione di algoritmi di ML** e strumenti di pre-processing dei
    dati
-   Software **open source** rilasciato sotto la GNU General Public
    License
-   Informazioni sul software: https://www.cs.waikato.ac.nz/ml/weka/
-   Weka Mailing, Documentazione
-   Accesso trasparente a toolbox noti come **scikit-learn**, **R**,
    **Deeplearning4j**

### Caratteristiche

-   Sviluppato in **Java**
-   Contiene:
    -   **Strumenti di pre-processing** dei dati (es. discretizzazione,
        selezione degli attributi)
    -   **Algoritmi di apprendimento** (Classificazione, Regressione,
        Clustering, Association Rules)
    -   **Interfaccia grafica** (include strumenti di visualizzazione
        dei dati)
    -   **Soluzioni per confrontare** algoritmi di apprendimento
-   **Facile da estendere** con nuovi/algoritmi ML aggiuntivi

------------------------------------------------------------------------

## 🎯 Uso di WEKA

-   Applicare algoritmi di apprendimento ai dataset e analisi
    dell'output
-   Diverse collezioni di dati disponibili:
    https://waikato.github.io/weka-wiki/datasets/
-   Include il **UCI Machine Learning Repository**:
    http://archive.ics.uci.edu/ml/
-   Confronto di diversi algoritmi di apprendimento
-   Sviluppo di nuovi metodi di apprendimento

------------------------------------------------------------------------

## 📥 Input: Applicare Algoritmi di Apprendimento

Gli algoritmi possono essere eseguiti: - Su dataset disponibili -
Chiamati all'interno del codice sorgente Java - Da ambienti come:
Octave/Matlab, R, Python e Hadoop

I dati di input sono **dataset**: - Un dataset è rappresentato come una
tabella in un Data Base (DB) - Composto da un insieme di **istanze** -
Ogni istanza è descritta da diversi **attributi**, ciascuno può essere
di tipo: - **Nominal** (insieme a una lista di valori predefiniti) -
**Numeric** (intero o reale) - **String** (lista di caratteri entro
`' '`) - **Date** - **Relational**

------------------------------------------------------------------------

## 📄 Formato ARFF

I dataset WEKA dovrebbero essere preferibilmente in **formato ARFF**.

Un file ARFF consiste di: - **Header** che descrive: - Nome del dataset
(o della relazione rappresentata) - Nome, tipo, valori per ogni
attributo che caratterizza il dataset - **Data Specification:** - Lista
di istanze - Per ogni istanza, i valori sono separati da virgola

Dettagli aggiuntivi sul formato ARFF:
https://waikato.github.io/weka-wiki/formats_and_processing/arff_syntax/

### Esempio di File ARFF

    % comment
    @relation heart-disease-simplified

    @attribute age numeric
    @attribute sex { female, male}
    @attribute chest_pain_type { typ_angina, asympt, non_anginal, atyp_angina}
    @attribute cholesterol numeric
    @attribute exercise_induced_angina { no, yes}
    @attribute class { present, not_present}

    @data
    63,male,typ_angina,233,no,not_present
    67,male,asympt,286,yes,present
    67,male,asympt,229,yes,present
    38,female,non_anginal,?,no,not_present
    ...

------------------------------------------------------------------------

## 🚀 Avvio di WEKA

Quando si avvia WEKA compaiono **cinque tab**:

``` mermaid
flowchart TB
    WEKA[WEKA] --> CLI[Simple CLI<br/>interfaccia a riga di comando]
    WEKA --> EXP[Explorer<br/>ambiente grafico per esplorazione e gestione dati]
    WEKA --> EX[Experimenter<br/>esperimenti e test statistici]
    WEKA --> KF[Knowledge Flow<br/>interfaccia 'data flow']
    WEKA --> WB[Workbench<br/>tutte le funzionalità in ambiente unificato]
```

-   **Simple CLI:** interfaccia a riga di comando
-   **Explorer:** ambiente grafico per l'esplorazione dei dati e la
    gestione dei dati
-   **Experimenter:** ambiente per eseguire esperimenti e test
    statistici usando diversi schemi di apprendimento
-   **Knowledge Flow:** interfaccia "data flow" (alternativa
    all'Experimenter)
    -   Da usare per dataset grandi
    -   Permette di comporre schemi di esperimenti
-   **Workbench:** tutte le funzionalità WEKA in un ambiente unificato

------------------------------------------------------------------------

## 🧪 Usare l'Experimenter

-   Permette il **confronto delle performance** di diversi algoritmi di
    apprendimento
    -   Caso di problemi di classificazione e regressione
-   I risultati possono essere **memorizzati** su un file o DB
-   **Opzioni disponibili:** cross-validation, learning curve, hold-out
-   È anche possibile considerare **diverse impostazioni dei parametri**

### Results Destination

-   Se il campo "file name" rimane vuoto viene creato un file temporaneo
    (nella directory TEMP)
-   Se l'utente vuole specificare un nome di file:
    -   **Browse** → scegli il percorso
    -   Specifica il nome del file
    -   Clicca **Save** → il nome del file appare nel campo
        corrispondente

### ARFF o CSV

-   **PRO:** possono essere creati senza richiedere classi aggiuntive
-   **CONS:**
    -   Un esperimento interrotto non può essere recuperato (es. a causa
        di un errore, o aggiunta di algoritmo o dataset)
    -   Funzionalità disponibile in caso di destinazione "JDBC Database"

### Experiment Type

Tre diversi tipi di esperimenti disponibili:

  -----------------------------------------------------------------------
  Tipo                   Descrizione
  ---------------------- ------------------------------------------------
  **Cross-validation**   Esegue cross-validation stratificata con il
  (default)              numero specificato di fold

  **Train/Test           Partiziona il dataset in training/test set
  Percentage Split**     secondo una data percentuale. La partizione è
  (data randomized)      eseguita dopo randomizzazione e stratificazione
                         (non è permesso specificare file separati per
                         training e test)

  **Train/Test           Partiziona il dataset in training e test set,
  Percentage Split**     data una percentuale, preservando l'ordine dei
  (order preserved)      dati
  -----------------------------------------------------------------------

### Datasets

-   **Scelta di un dataset specificando:**
    -   **Absolute Path:** clicca "Add new" → scegli il file dal
        percorso desiderato. Selezionando una directory, tutti i file
        ARFF al suo interno vengono aggiunti ricorsivamente
    -   **Relative Path:** adottato per facilitare l'esecuzione di
        esperimenti su macchine diverse. Spunta "Use relative paths"
        prima di cliccare "Add new"
-   I file possono essere **eliminati** dalla lista selezionandoli e
    cliccando "Delete Selected"
-   Oltre ai file ARFF, tutti i file che possono essere convertiti
    usando i "core converters" di Weka possono essere caricati
    -   Per vedere i formati supportati: apri il campo "file format"
    -   Es. ARFF (+ compresso), CSV, istanze serializzate binarie, XRFF
        (+ compresso)
-   **Di default**, l'attributo che rappresenta la classe è assunto
    essere l'**ultimo attributo**
    -   A meno che un formato dati (es. XRFF) contenga l'informazione
        sull'attributo di classe esplicitamente

### Iteration Control

Permette di specificare: - **Number of Repetitions:** numero di volte in
cui l'esperimento è ripetuto (es. numero di ripetizioni per la
cross-validation) - **Data sets first / Algorithms first:** se più di un
dataset e più di un algoritmo sono specificati, questa impostazione
fissa l'ordine di esecuzione - Es. prima tutti i dataset sono processati
con un dato algoritmo, poi si considera il prossimo algoritmo o
viceversa

### Algorithms

Scegli l'algoritmo cliccando su "Add New": - L'ultimo algoritmo usato
appare selezionato (o l'algoritmo ZeroR se la finestra è aperta per la
prima volta) - Clicca "chose" per scegliere l'algoritmo - Imposta i
parametri dell'algoritmo scelto poi clicca "Ok" per aggiungere
l'algoritmo alla lista degli algoritmi - Ripeti il processo se sei
interessato a diversi algoritmi

------------------------------------------------------------------------

## ▶️ Eseguire l'Esperimento

-   Clicca **Start** per avviare l'esperimento, date le impostazioni
    specificate
-   Se l'esperimento è stato definito correttamente, tre messaggi
    appaiono nel pannello Log:
    -   `10:33:04: Started`
    -   `13:41:15: Finished`
    -   `13:41:15: There were 0 errors`
-   Il risultato dell'esperimento è salvato nel file con il nome
    specificato all'inizio

------------------------------------------------------------------------

## 📊 Analizzare i Risultati

-   Per analizzare i risultati di **esperimenti precedenti**: clicca
    "File" → load il file contenente i risultati dell'esperimento
-   Clicca "Experiment" per analizzare il risultato dell'**ultimo
    esperimento eseguito**

``` mermaid
flowchart LR
    A[Source] --> B[Configure test]
    B --> C[Row key fields]
    B --> D[Column key fields]
    B --> E[Comparison field]
    B --> F[Significance]
    B --> G[Test base]
    B --> H[Perform test]
    H --> I[Result list]
```

-   **Source:** sorgente dei risultati (esperimento corrente o file
    caricato)
-   **Configure test:** configurazione del test
-   **Row key fields:** campi chiave di riga
-   **Column key fields:** campi chiave di colonna
-   **Comparison field:** campo di confronto
-   **Significance:** livello di significatività (es. 0.05)
-   **Test base:** base del test
-   **Show std. deviations:** mostra le deviazioni standard
-   **Perform test:** esegue il test
-   **Result list:** lista dei risultati

### Paired T-Test

L'Experimenter usa il **Paired T-Test** per confrontare le performance
degli algoritmi:

1.  Scegli il campo di confronto (es. `Percent_correct`)
2.  Seleziona l'algoritmo **baseline** (test base)
3.  Clicca **Perform test**
4.  I risultati mostrano:
    -   La percentuale di istanze classificate correttamente per ogni
        algoritmo
    -   Un \*\*asterisco (\*)** indica una differenza **statisticamente
        significativa\*\* rispetto alla baseline
    -   Il simbolo `v` o `*` indica se l'algoritmo è migliore o peggiore
        della baseline

``` mermaid
flowchart LR
    A[Seleziona Percent_correct] --> B[Seleziona baseline]
    B --> C[Perform test]
    C --> D[Risultati con significatività]
    D --> E[Identifica algoritmi migliori]
```

------------------------------------------------------------------------

## 📊 Statistiche e Output dell'Explorer

Quando si esegue un classificatore nell'**Explorer**, WEKA produce un
report dettagliato con diverse statistiche.

### Riepilogo Generale

-   **Correctly Classified Instances:** istanze classificate
    correttamente
-   **Incorrectly Classified Instances:** istanze classificate
    erroneamente
-   **Kappa statistic:** misura del miglioramento rispetto al predittore
    casuale
-   **Mean Absolute Error (MAE):** errore assoluto medio
-   **Root Mean Squared Error (RMSE):** errore quadratico medio
-   **Relative Absolute Error (RAE):** errore assoluto relativo
-   **Root Relative Squared Error (RRSE):** errore quadratico relativo

### Detailed Accuracy By Class

Il report **Detailed Accuracy By Class** mostra per ogni classe:

  ---------------------------------------------------------------------------------------------------
  Metrica                             Formula
  ----------------------------------- ---------------------------------------------------------------
  **TP Rate**                         $TP/(TP+FN)$

  **FP Rate**                         $FP/(FP+TN)$

  **Precision**                       $TP/(TP+FP)$

  **Recall**                          $TP/(TP+FN)$

  **F-Measure**                       $2 \times \frac{precision \times recall}{precision + recall}$
  ---------------------------------------------------------------------------------------------------

### Confusion Matrix

La **confusion matrix** è una matrice quadrata con: - **Classi reali**
sulle righe - **Classi predette** sulle colonne

``` mermaid
flowchart LR
    A[Classi reali<br/>righe] --> C[Confusion Matrix]
    B[Classi predette<br/>colonne] --> C
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 08_modelli_grafici.md ===== -->
```

------------------------------------------------------------------------

# PARTE III --- Modelli probabilistici e grafici

# 08 --- Probabilistic Graphical Models (PGM)

> *Riferimento: Bishop, PRML Ch. 8 · Koller & Friedman, Probabilistic
> Graphical Models*

------------------------------------------------------------------------

## 🧩 Cosa Sono i Modelli Grafici?

-   Sono **rappresentazioni diagrammatiche (grafi)** di distribuzioni di
    probabilità
-   Unione tra **teoria della probabilità** e **teoria dei grafi**
-   Chiamati anche **modelli grafici probabilistici**
-   Aumentano l'analisi invece di usare pura algebra
-   In un modello grafico probabilistico:
    -   Ogni **nodo** rappresenta una variabile casuale (o gruppo di
        variabili)
    -   I **collegamenti** esprimono relazioni probabilistiche tra
        variabili
-   **Strumento naturale** per gestire Incertezza e Complessità

``` mermaid
flowchart LR
    A[Probabilità] --> C[Modelli Grafici]
    B[Teoria dei Grafi] --> C
    C --> D[Gestione di Incertezza e Complessità]
```

------------------------------------------------------------------------

## 🔀 Direzionalità del Grafo

  ------------------------------------------------------------------------------------
                     Bayesian Networks                  Markov Networks
  ------------------ ---------------------------------- ------------------------------
  **Tipo**           Modelli grafici diretti            Modelli grafici non diretti

  **Collegamenti**   Frecce con direzione               Collegamenti senza frecce

  **Indipendenza**   Complessa (coinvolge la direzione  Definizione semplice
                     degli archi)                       

  **Relazioni**      Causali tra variabili              Vincoli (soft constraints) tra
                                                        variabili
  ------------------------------------------------------------------------------------

-   Due insiemi di nodi sono **condizionalmente indipendenti** dato un
    terzo insieme $C$ se tutti i nodi in $A$ e $B$ sono connessi
    attraverso nodi in $C$

------------------------------------------------------------------------

## 📐 Distribuzioni Congiunte e Condizionali

La teoria della probabilità necessaria può essere espressa in due
semplici equazioni:

### Sum Rule

La probabilità di una variabile si ottiene **marginalizzando**
(sommando) le altre variabili:

$$p(A) = \sum_b p(A, B)$$

### Product Rule

La probabilità congiunta è espressa in termini di condizionali:

$$p(A, B) = p(A \mid B) p(B) = p(B \mid A) p(A)$$

> Tutta l'inferenza e l'apprendimento probabilistico si riducono
> all'applicazione ripetuta delle regole di somma e prodotto.

------------------------------------------------------------------------

## 🔗 Dalla Distribuzione Congiunta al Modello Grafico

Data una distribuzione congiunta $p(A, B, C)$, per la product rule:

$$p(A, B, C) = p(A \mid B, C) p(B, C) = p(A \mid B, C) p(B \mid C) p(C)$$

-   Questa decomposizione vale per qualsiasi scelta della distribuzione
    congiunta
-   La fattorizzazione può essere rappresentata da un modello grafico
-   Si introduce un **nodo per ogni variabile casuale**
-   Diversi ordinamenti delle variabili darebbero un grafo diverso
-   Si associa ogni nodo con la sua **distribuzione condizionale**
-   Un grafo completamente connesso rappresenta tutte le dipendenze
    ottenute con la chain rule
-   **L'assenza di un collegamento rimuove una dipendenza**

------------------------------------------------------------------------

## 🧮 Teorema di Bayes

### Caso discreto

$$P_{Y \mid X}(y \mid x) = \frac{P_{XY}(x, y)}{P_X(x)} = \frac{P_{X \mid Y}(x \mid y) P_Y(y)}{\sum_{y' \in Val(Y)} P_{X \mid Y}(x \mid y') P_Y(y')}$$

### Caso continuo

$$f_{Y \mid x}(y \mid x) = \frac{f_{XY}(x, y)}{f_X(x)} = \frac{f_{X \mid Y}(x \mid y) f_Y(y)}{\int_{-\infty}^{+\infty} f_{X \mid Y}(x \mid y') f_Y(y') dy'}$$

-   **Prior probability:** $P_X(x)$ (o $f_X(x)$)
-   **Posterior probability:** $P_{X \mid Y}(x \mid y)$ (o
    $f_{X \mid Y}(x \mid y)$)

------------------------------------------------------------------------

## 🌳 Reti Bayesiane

Consideriamo un'arbitraria distribuzione congiunta $p(a, b, c)$ definita
su tre variabili $a$, $b$ e $c$. Applicando la regola del prodotto:

$$p(a, b, c) = p(c \mid a, b) p(a, b) = p(c \mid a, b) p(b \mid a) p(a)$$

### Trasformazione in Modello Grafico

1.  Introduciamo un **nodo per ogni variabile**
2.  Per ogni distribuzione condizionale aggiungiamo un **collegamento
    diretto** dai nodi corrispondenti alle variabili su cui la
    distribuzione è condizionata
3.  Associamo ogni nodo con la corrispondente probabilità condizionale
4.  Es. per il fattore $p(c \mid a, b)$ inseriamo un collegamento dai
    nodi $a$ e $b$ al nodo $c$
5.  Se esiste un collegamento da $a$ a $b$, chiamiamo $a$ **parent** del
    nodo $b$, e $b$ **child** del nodo $a$

``` mermaid
flowchart LR
    A[a] --> C[c]
    B[b] --> C[c]
    A --> B
```

------------------------------------------------------------------------

## 📦 Fattorizzazione

In generale la distribuzione congiunta definita da un grafo di $k$ nodi
è:

$$p(x) = \prod_{i=1}^{k} p(x_i \mid pa_i)$$

dove $pa_i$ indica l'insieme dei parent di $x_i$, e
$x = \{x_1, x_2, \dots, x_k\}$.

-   L'equazione esprime la proprietà di **fattorizzazione** della
    distribuzione congiunta di un modello grafico
-   I grafi devono essere **grafi diretti aciclici** (DAG)

### Chain Rule for BN

Sia $G$ un grafo di una rete Bayesiana definito sulle variabili
$X_1, \dots, X_n$. Diciamo che la distribuzione $P_B$ si fattorizza
rispetto a $G$ se $P_B$ può essere espressa come prodotto delle CPD.

> Una **rete Bayesiana** è una coppia $(G, \theta_G)$ dove $P_B$ si
> fattorizza su $G$, e dove $P_B$ è specificata da un insieme di **CPD**
> (Conditional Probability Distributions), denotato con $\theta_G$,
> associate ai nodi di $G$.

------------------------------------------------------------------------

## 🔗 Indipendenza Condizionale

Consideriamo tre variabili $a$, $b$ e $c$, e supponiamo che la
distribuzione condizionale di $a$, dati $b$ e $c$, non dipenda da $b$:

$$p(a \mid b, c) = p(a \mid c)$$

Possiamo esprimere il concetto in modo diverso:

$$p(a, b \mid c) = p(a \mid b, c) p(b \mid c) = p(a \mid c) p(b \mid c)$$

ovvero, $a$ e $b$ sono **statisticamente indipendenti, dato $c$**. In
notazione:

$$a \perp b \mid c$$

> L'indipendenza condizionale gioca un ruolo importante nei modelli
> probabilistici: semplifica la struttura del modello e i calcoli
> richiesti per l'inferenza e l'apprendimento.

### Assunzioni Locali di Markov

In una rete Bayesiana, ogni variabile è condizionalmente indipendente
dai suoi **non-discendenti** dati i suoi **parent**:

$$X_i \perp \text{NonDesc}(X_i) \mid \text{Pa}(X_i)$$

Questa è chiamata **assunzione locale di Markov** ed è la base per la
fattorizzazione della distribuzione congiunta.

------------------------------------------------------------------------

## 🚫 D-Separation

Consideriamo un grafo diretto in cui $A$, $B$ e $C$ sono insiemi di nodi
non intersecanti. Vogliamo stabilire se si verifica una condizione di
indipendenza $A \perp B \mid C$.

Consideriamo tutti i possibili **percorsi** da ogni nodo in $A$ ad ogni
nodo in $B$. Uno di questi path si dice **blocked** se include un nodo
tale che: - Il nodo ha **frecce convergenti** nel percorso
($\rightarrow W \leftarrow$) e il nodo né esso né nessuno dei suoi
discendenti è nell'insieme $C$, **oppure** - Il nodo **non ha frecce
convergenti** nel percorso ($\rightarrow W \rightarrow$ o
$\leftarrow W \rightarrow$) e il nodo è nell'insieme $C$ (osservato)

Se **tutti i path sono bloccati** allora si dice che $C$ **d-separa**
$A$ da $B$.

$$A \perp B \mid C \quad \text{se } C \text{ d-separa } A \text{ da } B$$

------------------------------------------------------------------------

## 🧩 Markov Random Fields

-   I modelli grafici diretti specificano una fattorizzazione della
    distribuzione congiunta con un prodotto di distribuzioni
    condizionali locali
-   Definiscono un insieme di proprietà di indipendenza condizionale
    soddisfatte da una distribuzione che si fattorizza in base al grafo

Un **Markov Random Field** (o Markov Network o modello grafico non
diretto) è costituito da: - Un insieme di nodi, ognuno dei quali
corrisponde a una variabile o a un gruppo di variabili - Un insieme di
collegamenti non diretti che connettono una coppia di nodi

### Funzioni Potenziali

Poiché le funzioni potenziali devono essere **strettamente positive**, è
conveniente esprimerle come esponenziali:

$$\psi(x_C) = \exp\{-E(x_C)\}$$

dove $E(x_C)$ è detta **energy function**, e la rappresentazione
esponenziale è detta **distribuzione di Boltzmann**.

-   A differenza dei fattori nei modelli diretti, i potenziali in un
    modello indiretto **non hanno una specifica interpretazione
    probabilistica**
-   L'approccio è più flessibile, non dobbiamo rispettare un vincolo di
    normalizzazione

------------------------------------------------------------------------

## 🔍 Inferenza nei Modelli Grafici

**Problema:** calcolare la distribuzione a posteriori di uno o più
sottoinsiemi di nodi osservando i valori di altri nodi.

Consideriamo il grafo non diretto (catena). La distribuzione congiunta
prende la forma:

$$p(x) = \frac{1}{Z} \psi_{1,2}(x_1, x_2) \psi_{2,3}(x_2, x_3) \cdots \psi_{N-1,N}(x_{N-1}, x_N)$$

-   Se gli $N$ nodi rappresentano variabili discrete aventi $K$ stati,
    ogni funzione potenziale $\psi_{n-1,n}(x_{n-1}, x_n)$ rappresenta
    una tabella $K \times K$
-   La distribuzione congiunta ha $(N-1)K^2$ parametri

### Variable Elimination

L'idea di base: eliminare le variabili una alla volta,
marginalizzandole, per calcolare l'inferenza in modo efficiente.

-   Riduce la complessità computazionale da **esponenziale a lineare**
    lungo sequenze di nodi
-   Crea **fattori intermedi** ($\tau$) durante l'eliminazione
    progressiva delle variabili

### Clique Tree / Junction Tree Message Passing

L'algoritmo **Sum-Product** è guidato dalla struttura ad albero delle
cricche (*clique tree*).

**Operazioni per cricca:** 1. **Ricezione** dei messaggi entranti dalle
cricche adiacenti 2. **Moltiplicazione** dei messaggi con i fattori
locali 3. **Marginalizzazione/Somma** sulle variabili 4. **Invio** del
messaggio alla cricca adiacente

``` mermaid
flowchart LR
    A[Cricca 1] -->|messaggio| B[Cricca 2]
    B -->|messaggio| C[Cricca 3]
    B -->|messaggio| D[Cricca 4]
```

------------------------------------------------------------------------

## 🎲 Sampling (Inferenza Approssimata)

Molti algoritmi di ML si basano sul **campionamento** da una
distribuzione di probabilità e sull'uso di questi campioni per formare
una **stima Monte Carlo** di una quantità desiderata.

-   I PGM sono **modelli generativi**
-   Il sampling fornisce un modo flessibile per approssimare molte somme
    e integrali a costo ridotto
-   A volte per velocizzare una somma costosa ma trattabile (es.
    subsample del costo di training con minibatches)
-   In altri casi, gli algoritmi di apprendimento richiedono di
    approssimare una somma o integrale intrattabile (es. gradiente della
    log partition function di un modello non diretto)

### Forward Sampling

-   Campiona le variabili nell'ordine topologico del grafo
-   Basato sul **Hoeffding bound** per garantire l'accuratezza

### Ancestral Sampling

L'**ancestral sampling** è il metodo per generare campioni da una rete
Bayesiana: 1. Ordina le variabili in **ordine topologico** (i parent
prima dei figli) 2. Per ogni variabile $X_i$ in ordine: - Campiona
$x_i \sim p(x_i \mid pa_i)$ usando i valori già campionati dei parent 3.
Il campione risultante $(x_1, \dots, x_n)$ è un campione valido dalla
distribuzione congiunta

``` mermaid
flowchart LR
    A[Ordina variabili topologicamente] --> B[Per ogni variabile X_i]
    B --> C[Campiona x_i ~ p(x_i | pa_i)]
    C --> D{Tutte le variabili?}
    D -->|No| B
    D -->|Sì| E[Campione valido dalla congiunta]
```

### Rejection Sampling

Un task più difficile è calcolare probabilità condizionali come
$P(y \mid E = e)$: - Genera campionamenti di $x$ da $P(X)$ e **rigetta**
tutti quelli non compatibili con $e$ - Quanti campionamenti dobbiamo
fare se $P(e) = 0.001$ per avere $M^*$ campioni non rigettati?

$$M = M^* / P(e)$$

### Likelihood Weighting

**Intuizione:** invece di rigettare a posteriori, **forziamo** il
campionamento ad assumere valori appropriati rispetto a quelli noti: -
Quando campioniamo un nodo $X_i$ i cui valori sono stati osservati, lo
impostiamo ai suoi valori noti - Ad ogni campionamento è associato un
**peso**

$$\tilde{P}_{\mathcal{D}}(y \mid e) = \frac{\sum_{m=1}^{M} w[m] \mathbb{1}\{y[m] = y\}}{\sum_{m=1}^{M} w[m]}$$

### Gibbs Sampling

Il Gibbs sampling è un esempio di algoritmo **Markov chain Monte Carlo
(MCMC)**: - In un algoritmo MCMC i campionamenti si ottengono con una
catena di Markov la cui distribuzione stazionaria è quella desiderata
$p(x)$ - Lo stato della catena di Markov è l'insieme degli assegnamenti
dei valori ad ogni variabile

La catena di Markov per il Gibbs sampler viene costruita come segue: 1.
Ad ogni passo viene selezionata una variabile $X_i$ (a caso) 2. Si
calcola la distribuzione condizionale $p(x_i \mid x_{U \setminus i})$ 3.
Si sceglie un valore $x_i$ da tale distribuzione 4. Il campione $x_i$
sostituisce il precedente valore della $i$-esima variabile

$$p(x_i \mid x_{U \setminus i}) = \frac{\prod_{C \in \mathcal{E}_i} \psi_C(x_C)}{\sum_{x_i} \prod_{C \in \mathcal{E}_i} \psi_C(x_C)}$$

dove $\mathcal{E}_i$ indica l'insieme delle cricche contenenti l'indice
$i$.

### Metropolis-Hastings (MCMC)

Il **Metropolis-Hastings** è un algoritmo MCMC che genera una **proposta
di stato** $\tilde{x}$ e la accetta con una certa probabilità:

1.  Genera una proposta $\tilde{x}$ da una distribuzione di proposta
    $q(\tilde{x} \mid x)$
2.  Calcola la probabilità di accettazione:

$$\alpha = \min\left(1, \frac{\prod \psi(\tilde{x})}{\prod \psi(x)}\right)$$

3.  Accetta $\tilde{x}$ con probabilità $\alpha$, altrimenti mantieni
    $x$

### Importance Sampling

L'**importance sampling** campiona da una **distribuzione di proposta**
$q(x)$ semplice e **ripesa** i campioni:

-   Ogni campione $x^{(t)}$ è pesato dal rapporto
    $\frac{p(x^{(t)})}{q(x^{(t)})}$
-   La stima del valore atteso è:

$$E[f(x)] \approx \frac{1}{T} \sum_{t=1}^{T} \frac{p(x^{(t)})}{q(x^{(t)})} f(x^{(t)})$$

-   Utile quando $p(x)$ è difficile da campionare direttamente

------------------------------------------------------------------------

## 📉 Metodi Variazionali

L'idea degli approcci variazionali sta nel **convertire il problema
dell'inferenza probabilistica in un problema di ottimizzazione**.

-   L'approccio di base è simile all'importance sampling, ma invece di
    scegliere una singola $q(x)$ a priori, viene utilizzata una
    **famiglia di distribuzioni** $\{q(x)\}$, e l'obiettivo di
    ottimizzazione è scegliere un elemento da tale famiglia

Definiamo l'energia di una configurazione $x$ come:

$$E(x) = -\log p(x) - \log Z$$

e l'energia libera variazionale come:

$$F(\{q(x)\}) = \sum_x q(x) E(x) + \sum_x q(x) \log q(x) = -\sum_x q(x) \log p(x) + \sum_x q(x) \log q(x) - \log Z$$

### Mean Field Approximation

In questo caso $\{q(x)\}$ corrisponde alla famiglia delle distribuzioni
**fattorizzate**:

$$q(x) = \prod_i q_i(x_i)$$

------------------------------------------------------------------------

## 📈 Apprendimento dei Parametri

### Maximum Likelihood: Dati Completi

Dati un insieme di dati $d$ di $N$ osservazioni indipendenti ed
identicamente distribuite di tutte le variabili per un modello grafico
diretto:

$$d = \{x^{(1)}, \dots, x^{(N)}\}, \quad \text{dove } x^{(n)} = (x^{(n)}_1, \dots, x^{(n)}_d)$$

La likelihood è definita come:

$$p(d \mid \theta) = \prod_{n=1}^{N} p(x^{(n)} \mid \theta)$$

Vogliamo trovare i parametri che massimizzano la likelihood, il che
equivale a massimizzare la log-likelihood:

$$\mathcal{L}(\theta) = \log p(d \mid \theta) = \sum_{n=1}^{N} \log p(x^{(n)} \mid \theta) = \sum_{n=1}^{N} \sum_{i=1}^{d} \log p(x^{(n)}_i \mid x^{(n)}_{Pa(i)}, \theta_i)$$

Se assumiamo che i parametri $\theta_i$ sono distinti e indipendenti,
allora:

$$\mathcal{L}(\theta) = \sum_{i=1}^{d} \mathcal{L}_i(\theta_i)$$

### EM (Expectation Maximization)

L'algoritmo di expectation maximization si alterna fra il massimizzare
$F$ rispetto a $q$ e $\theta$, rispettivamente, lasciando fisso l'altro
parametro.

Partendo con un parametro iniziale $\theta_0$, la $(k+1)$-esima
iterazione consiste nei seguenti due passi:

$$E \text{ step: } q_{[k+1]} \leftarrow \arg\max_q \mathcal{F}(q, \theta_{[k]})$$
$$M \text{ step: } \theta_{[k+1]} \leftarrow \arg\max_\theta \mathcal{F}(q_{[k+1]}, \theta)$$

-   Il massimo nel passo E si ottiene impostando
    $q_{[k+1]} = p(y \mid x, \theta_{[k]})$, punto per il quale si
    ottiene
    $\mathcal{F}(q_{[k+1]}, \theta_{[k]}) = \mathcal{L}(\theta_{[k]})$

### Stima MAP dei Parametri

La **stima MAP** (Maximum A Posteriori) massimizza la posterior,
aggiungendo un **prior** come regolarizzatore:

$$L'(\theta) = \sum_i L_i(\theta_i) + \log p(\theta_i)$$

-   Il termine $\log p(\theta_i)$ funge da **regolarizzatore** per
    limitare l'overfitting quando i dati sono insufficienti

### Variabili Nascoste e Algoritmo EM

Con variabili nascoste, la log-likelihood incompleta è:

$$L(\theta) = \log \sum_y p(x, y \mid \theta)$$

-   Si usa la **Jensen's inequality** e il funzionale di energia
    $F(q, \theta) = E_\theta[\log q] + H_\theta(q)$
-   Alternanza tra **E-step**
    ($q^{[k+1]} \leftarrow p(y \mid x, \theta^{[k]})$) e **M-step**
    ($\theta^{[k+1]} \leftarrow \arg\max_\theta F$)

### Apprendimento Bayesiano dei Parametri (Dati Completi)

-   Si usa la distribuzione di **Dirichlet** come prior coniugata alla
    distribuzione multinomiale:

$$P(\theta) = \text{Dir}(\alpha_1, \dots, \alpha_K)$$

-   La posterior è una nuova Dirichlet con iperparametri aggiornati:

$$\tilde{\alpha}_{ijk} = \alpha_{ijk} + n_{ijk}$$

(somma delle prior con le conte effettive)

### Apprendimento Bayesiano dei Parametri (Dati Incompleti)

-   La distribuzione a posteriori diventa una **mixture di Dirichlet**
    con un numero esponenziale di termini
-   Soluzioni adottate: **completamento di Viterbi**, metodi **MCMC** o
    **Variational Bayesian**

------------------------------------------------------------------------

## 🏗️ Structure Learning

Dato un dataset di osservazioni di $(A, B, C, D, E)$ è possibile
apprendere la **struttura** del modello grafico? ($m$ denota la
struttura del grafo = l'insieme degli archi)

  -----------------------------------------------------------------------
  Approccio                        Descrizione
  -------------------------------- --------------------------------------
  **Constraint-Based Learning**    Usa test statistici di indipendenza
                                   marginale e condizionale. Trova
                                   l'insieme dei DAG in cui le relazioni
                                   di d-separation sono confrontabili con
                                   i risultati dei test di indipendenza
                                   condizionale

  **Score-Based Learning**         Usa uno score globale come il **BIC
                                   score** o la likelihood marginale
                                   Bayesiana. Trova le strutture che
                                   massimizzano tale score
  -----------------------------------------------------------------------

### Bayesian Marginal Likelihood Score

Lo **score bayesiano** usa la **likelihood marginale** in forma chiusa:

$$\text{score}(m) = \log P(D \mid m) = \int P(D \mid \theta, m) P(\theta \mid m) d\theta$$

-   Si usa la funzione **Gamma** ($\Gamma$) applicata agli iperparametri
    di Dirichlet $\alpha_{ijk}$ e alle conte dei dati $n_{ijk}$
-   Lo score si **scompone per singolo nodo** $i$ e si aggiunge la prior
    sulla struttura:

$$\text{score}(m) = \log P(D \mid m) + \log P(m)$$

### Strategie di Ricerca

-   **Greedy Search:** modifica la struttura con modifiche elementari
    (aggiunta, eliminazione, inversione di archi) e mantiene la modifica
    che migliora lo score
-   **Bayesian Structural EM** (Friedman): apprende la struttura in
    presenza di **dati incompleti**, alternando stima dei parametri e
    ricerca della struttura

``` mermaid
flowchart LR
    A[Struttura iniziale] --> B[Modifica elementare<br/>aggiungi/elimina/inverti arco]
    B --> C{Score migliorato?}
    C -->|Sì| D[Mantieni modifica]
    C -->|No| E[Scarta modifica]
    D --> B
    E --> B
```

------------------------------------------------------------------------

## 🔗 Conditional Random Fields (CRF)

-   $Y$: variabili target, $X$: variabili osservate
-   Un CRF è un modello grafico **non diretto** i cui nodi corrispondono
    a $Y \cup X$
-   Piuttosto che codificare la distribuzione $P(Y, X)$ si vuole
    rappresentare la distribuzione **condizionata** $P(Y \mid X)$
-   Non permettiamo che ci siano potenziali fra le variabili $X$
-   La rete è annotata con un insieme di fattori
    $\phi(D_1), \dots, \phi(D_m)$, $D_i \not\subseteq X$
-   La rete codifica la seguente distribuzione:

$$P(Y \mid X) = \frac{1}{Z(X)} \hat{P}(Y, X)$$

$$\hat{P}(Y, X) = \prod_{i=1}^{m} \phi_i(D_i)$$

$$Z(X) = \sum_Y \hat{P}(Y, X)$$

------------------------------------------------------------------------

## 📋 Plate Models

I **plate models** sono una notazione compatta per rappresentare modelli
grafici con **variabili ripetute** (es. osservazioni i.i.d.). Un "plate"
(rettangolo) indica che la struttura al suo interno viene replicata più
volte.

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 09_alberi_e_regole.md ===== -->
```

------------------------------------------------------------------------

# PARTE IV --- Alberi, ensemble e kernel methods

# 09 --- Alberi di Decisione e Regole

> *Riferimento: Witten & Frank, Data Mining, Ch. 3, 4*

------------------------------------------------------------------------

## 🌳 Alberi di Decisione

### Strategia: Divide-and-Conquer

Strategia: apprendimento **top-down** usando un processo ricorsivo di
divide-and-conquer:

1.  **Prima:** seleziona l'attributo per il nodo radice. Crea un ramo
    per ogni possibile valore dell'attributo
2.  **Poi:** dividi le istanze in sottoinsiemi, uno per ogni ramo che si
    estende dal nodo
3.  **Infine:** ripeti ricorsivamente per ogni ramo, usando solo le
    istanze che raggiungono il ramo

**Stop** se tutte le istanze hanno la stessa classe.

``` mermaid
flowchart TB
    A[Seleziona attributo per la radice] --> B[Crea ramo per ogni valore]
    B --> C[Dividi istanze in sottoinsiemi]
    C --> D{Ripeti ricorsivamente}
    D -->|Tutte stessa classe| E[Stop]
    D -->|No| A
```

------------------------------------------------------------------------

## 🎯 Quale Attributo Selezionare?

-   Vogliamo ottenere l'**albero più piccolo**
-   **Euristica:** scegli l'attributo che produce i nodi "più puri"
-   **Criterio di selezione popolare:** information gain
-   L'information gain aumenta con la purezza media dei sottoinsiemi
-   **Strategia:** tra gli attributi disponibili per lo split, scegli
    quello che dà il maggiore information gain
-   L'information gain richiede una misura di impurità
-   **Misura di impurità adottata:** l'**entropia** della distribuzione
    di classe (misura della teoria dell'informazione)

### Calcolare l'Informazione

Abbiamo una distribuzione di probabilità: la distribuzione di classe in
un sottoinsieme di istanze.

L'informazione attesa richiesta per determinare un esito (cioè il valore
di classe) è l'**entropia** della distribuzione:

$$\text{Entropy}(p_1, p_2, \dots, p_n) = -p_1 \log p_1 - p_2 \log p_2 \dots - p_n \log p_n$$

-   Usando logaritmi in base 2, l'entropia dà l'informazione richiesta
    in **bit attesi**
-   L'entropia è **massima** quando tutte le classi sono ugualmente
    probabili e **minima** quando una delle classi ha probabilità 1

### Calcolare l'Information Gain

**Information gain** = informazione prima dello split − informazione
dopo lo split

**Esempio dai dati meteo:**

    Gain(Outlook) = Info([9,5]) - info([2,3],[4,0],[3,2]) = 0.940 - 0.693 = 0.247 bits

    Gain(Outlook)     = 0.247 bits
    Gain(Temperature) = 0.029 bits
    Gain(Humidity)    = 0.152 bits
    Gain(Windy)       = 0.048 bits

------------------------------------------------------------------------

## ⚠️ Attributi Molto Ramificati

**Problematici:** attributi con un grande numero di valori (caso
estremo: ID code)

-   I sottoinsiemi sono più probabili di essere puri se c'è un grande
    numero di valori
-   L'information gain è **sbilanciato** verso la scelta di attributi
    con un grande numero di valori
-   Questo può risultare in **overfitting** (selezione di un attributo
    non ottimale per la predizione)
-   Un problema aggiuntivo negli alberi di decisione è la **data
    fragmentation**

### Gain Ratio

-   Il **gain ratio** è una modifica dell'information gain che riduce il
    suo bias verso attributi con molti valori
-   Il gain ratio tiene conto del **numero e della dimensione dei rami**
    quando sceglie un attributo
-   Corregge l'information gain prendendo in considerazione
    l'**informazione intrinseca** di uno split
-   **Informazione intrinseca:** entropia della distribuzione delle
    istanze nei rami
-   Misura quanta informazione è necessaria per dire a quale ramo
    appartiene un'istanza scelta casualmente

**Formula del gain ratio:**

$$\text{GainRatio}(A) = \frac{\text{Gain}(A)}{\text{IntrinsicInfo}(A)}$$

dove l'informazione intrinseca è:

$$\text{IntrinsicInfo}(A) = -\sum_{i=1}^{v} \frac{|S_i|}{|S|} \log_2 \frac{|S_i|}{|S|}$$

con $v$ il numero di valori dell'attributo $A$ e $S_i$ il sottoinsieme
di istanze con il valore $i$-esimo.

------------------------------------------------------------------------

## 💬 Discussione: ID3

La **top-down induction of decision trees (ID3)** è l'algoritmo classico
per costruire alberi di decisione: - Usa l'information gain per
selezionare gli attributi - Cresce l'albero ricorsivamente con
divide-and-conquer - È la base per algoritmi più avanzati come C4.5 e
CART

**Limiti di ID3:** - Bias verso attributi con molti valori (risolto dal
gain ratio in C4.5) - Non gestisce direttamente attributi numerici
(richiede discretizzazione) - Non gestisce direttamente valori mancanti

------------------------------------------------------------------------

## ✂️ Pruning

Per prevenire l'overfitting sui dati di training: **"potare"** l'albero
di decisione.

### Due Strategie

  -----------------------------------------------------------------------
  Strategia                        Descrizione
  -------------------------------- --------------------------------------
  **Postpruning**                  Prende un albero di decisione
                                   completamente cresciuto e scarta le
                                   parti inaffidabili

  **Prepruning**                   Ferma la crescita di un ramo quando
                                   l'informazione diventa inaffidabile
  -----------------------------------------------------------------------

> **Postpruning preferito in pratica** --- il prepruning può "fermarsi
> presto".

------------------------------------------------------------------------

## � Predizione Numerica tramite Alberi

Per la **predizione numerica** (regressione), gli alberi di decisione
vengono adattati:

### Regression Trees

Un **regression tree** è un "albero di decisione" dove ogni foglia
predice una **quantità numerica**: - Il valore predetto è il **valore
medio** delle istanze di training che raggiungono la foglia - La misura
di impurità per la selezione degli attributi è basata sulla **riduzione
della varianza** (o errore quadratico) invece che sull'entropia

### Model Trees

Un **model tree** è un "regression tree" con **modelli di regressione
lineare** ai nodi foglia: - Le foglie contengono un modello lineare (es.
$y = w_0 + w_1 x_1 + \dots$) invece di una semplice media - **Patch
lineari** approssimano la funzione continua - Più accurati dei
regression tree semplici perché catturano relazioni lineari locali

``` mermaid
flowchart TB
    A[Predizione numerica con alberi] --> B[Regression Tree<br/>foglia = valore medio]
    A --> C[Model Tree<br/>foglia = modello lineare]
    B --> D[Impurità: riduzione varianza]
    C --> E[Patch lineari approssimano la funzione]
```

------------------------------------------------------------------------

## �📜 Regole di Classificazione

### Inferire Regole Rudimentali (1R)

Il **1R rule learner** apprende un albero di decisione a 1 livello: - Un
insieme di regole che testano tutte un particolare attributo
identificato come quello che produce il **minore errore di
classificazione**

**Versione base** per trovare l'insieme di regole da un dato training
set (assume attributi nominali): - Per ogni attributo: - Fai un ramo per
ogni valore dell'attributo - A ogni ramo, assegna il valore di classe
più frequente delle istanze appartenenti a quel ramo - Error rate:
proporzione di istanze che non appartengono alla classe di maggioranza
del loro ramo corrispondente - Scegli l'attributo con il **minore error
rate**

------------------------------------------------------------------------

## 🧩 Covering Algorithms

-   Un albero di decisione può essere convertito in un insieme di regole
-   Semplice, ma l'insieme di regole è **eccessivamente complesso**
-   Conversioni più efficaci non sono banali e possono richiedere molta
    computazione
-   Invece, possiamo generare l'insieme di regole **direttamente**
-   **Un approccio:** per ogni classe a turno, trova l'insieme di regole
    che copre tutte le istanze in essa (escludendo le istanze non nella
    classe)

Chiamato **approccio covering** (chiamato anche approccio
**separate-and-conquer**): - A ogni stadio dell'algoritmo, una regola
viene identificata che "copre" alcune delle istanze

### Semplice Covering Algorithm

**Idea di base:** genera una regola aggiungendo test che massimizzano
l'accuratezza della regola.

-   Simile alla situazione negli alberi di decisione: problema di
    selezionare un attributo su cui fare lo split
-   Ma: l'induttore di alberi di decisione massimizza la purezza
    complessiva
-   Ogni nuovo test **riduce la copertura** della regola

### Pseudo-codice per PRISM

    For each class C
        Initialize E to the instance set
        While E contains instances in class C
            Create a rule R with an empty left-hand side that predicts class C
            Until R is perfect (or there are no more attributes to use) do
                For each attribute A not mentioned in R, and each value v,
                    Consider adding the condition A = v to the left-hand side of R
                    Select A and v to maximize the accuracy p/t
                    (break ties by choosing the condition with the largest p)
                Add A = v to R
            Remove the instances covered by R from E

### Separate and Conquer Rule Learning

I metodi di apprendimento di regole come quello impiegato da PRISM (per
ogni classe) sono chiamati algoritmi **separate-and-conquer**:

1.  **Prima,** identifica una regola utile
2.  **Poi,** separa tutte le istanze che copre
3.  **Infine,** "conquista" le istanze rimanenti

**Differenza rispetto ai metodi divide-and-conquer:** - Il sottoinsieme
coperto da una regola non deve essere esplorato ulteriormente

``` mermaid
flowchart LR
    A[Identifica regola utile] --> B[Separa istanze coperte]
    B --> C[Conquista istanze rimanenti]
    C --> D{Ripeti}
    D -->|Sì| A
    D -->|No| E[Fine]
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 10_kernel_svm.md ===== -->
```
# 13 --- Boosting e Ensemble Learning

> *Riferimento: Murphy, MLPP Ch. 16*

------------------------------------------------------------------------

## 🧩 Ensemble Learning

L'**ensemble learning** si riferisce all'apprendimento di una
**combinazione pesata di modelli base** della forma:

$$f(y \mid x, \pi) = \sum_{m \in \mathcal{M}} w_m f_m(y \mid x)$$

dove i $w_m$ sono parametri regolabili.

-   L'ensemble learning è a volte chiamato **metodo del comitato**,
    poiché ogni modello base $f_m$ ottiene un **voto pesato**
-   Chiaramente l'ensemble learning è strettamente correlato
    all'apprendimento di modelli a **adaptive basis function**
-   In effetti, si può sostenere che una **rete neurale** è un metodo
    ensemble, dove $f_m$ rappresenta l'$m$-esima hidden unit, e $w_m$
    sono i pesi dello strato di output
-   Possiamo anche pensare al **boosting** come a un tipo di ensemble
    learning, dove i pesi sui modelli base sono determinati
    **sequenzialmente**

------------------------------------------------------------------------

## 🌳 Alberi di Decisione come ABM

Possiamo scrivere il modello di un albero di decisione nella seguente
forma:

$$f(x) = E[y \mid x] = \sum_{m=1}^{M} w_m \mathbb{I}(x \in R_m) = \sum_{m=1}^{M} w_m \phi(x; v_m)$$

dove $R_m$ è l'$m$-esima regione, $w_m$ è la risposta media in questa
regione, e $v_m$ codifica la scelta della variabile su cui fare lo
split, e il valore di soglia, sul percorso dalla radice all'$m$-esima
foglia.

> Questo rende chiaro che un modello **CART** (classification and
> regression tree) è solo un modello a adaptive basis-function, dove le
> basis functions definiscono le regioni, e i pesi specificano il valore
> di risposta in ogni regione.

### Svantaggio Principale di CART

Il principale svantaggio di CART è che **non predicono in modo molto
accurato**: - Questo è dovuto alla natura **greedy** della costruzione
dell'albero - Gli alberi sono **instabili**: piccoli cambiamenti ai dati
di input possono avere grandi effetti sulla struttura dell'albero,
causando errori in cima che influenzano il resto dell'albero - Diciamo
che gli alberi sono **stimatori ad alta varianza**

------------------------------------------------------------------------

## 🌲 Random Forests

Un modo per ridurre la varianza di una stima è **mediare insieme molte
stime**: - Possiamo addestrare $M$ alberi diversi su diversi
sottoinsiemi dei dati, scelti casualmente con sostituzione, e poi
calcolare l'ensemble:

$$f(x) = \sum_{m=1}^{M} \frac{1}{M} f_m(x)$$

dove $f_m$ è l'$m$-esimo albero. Questa tecnica è chiamata **bagging**
(bootstrap aggregating).

> Sfortunatamente, semplicemente rieseguire lo stesso algoritmo di
> apprendimento su diversi sottoinsiemi dei dati può risultare in
> predittori **altamente correlati**, il che limita la quantità di
> riduzione della varianza possibile.

### Decorrelare i Base Learners

La tecnica nota come **random forests** tenta di **decorrelare** i base
learners apprendendo alberi basati su un sottoinsieme scelto casualmente
di variabili di input, così come un sottoinsieme scelto casualmente di
casi di dati.

-   La ragione di ciò è la **correlazione** degli alberi in un campione
    bootstrap ordinario: se una o poche feature sono predittori molto
    forti per la variabile di risposta, queste feature saranno
    selezionate in molti degli $B$ alberi, causando la loro correlazione
-   Tipicamente, per un problema di classificazione con $p$ feature,
    $\sqrt{p}$ (arrotondato per difetto) feature sono usate in ogni
    split
-   Per problemi di regressione gli inventori raccomandano $p/3$
    (arrotondato per difetto) con una dimensione minima del nodo di 5
    come default
-   In pratica i migliori valori per questi parametri dipenderanno dal
    problema, e dovrebbero essere trattati come **tuning parameters**

------------------------------------------------------------------------

## ⚡ Extremely Randomized Trees (ExtraTrees)

Aggiungere un ulteriore passo di randomizzazione produce **extremely
randomized trees**, o **ExtraTrees**.

Sebbene simili alle ordinary random forests in quanto sono un ensemble
di alberi individuali, ci sono **due differenze principali**: 1.
**Prima:** ogni albero è addestrato usando **l'intero campione di
apprendimento** (piuttosto che un campione bootstrap) 2. **Seconda:** lo
split top-down nel tree learner è **randomizzato**. Invece di calcolare
il cut-point localmente ottimale per ogni feature considerata (basato
su, es., information gain o Gini impurity), viene selezionato un
**cut-point casuale** - Questo valore è selezionato da una distribuzione
uniforme entro il range empirico della feature (nel training set
dell'albero). Poi, di tutti gli split generati casualmente, lo split che
produce il punteggio più alto è scelto per dividere il nodo

Simile alle ordinary random forests, il numero di feature selezionate
casualmente da considerare a ogni nodo può essere specificato.

------------------------------------------------------------------------

## 🚀 Boosting

Il **boosting** è un algoritmo greedy per adattare modelli a adaptive
basis-function, dove i $\phi_m$ sono generati da un algoritmo chiamato
**weak learner** o **base learner**.

-   L'algoritmo funziona applicando il weak learner **sequenzialmente**
    a versioni pesate dei dati, dove più peso è dato agli esempi che
    sono stati **misclassificati** nei round precedenti
-   L'obiettivo del boosting è risolvere il seguente problema di
    ottimizzazione:

$$\min_f \sum_{i=1}^{N} L(y_i, f(x_i))$$

### Funzioni di Loss Comuni

  ---------------------------------------------------------------------------------------------------------------------------------
  Nome          Loss                           Derivata                     Minimizzatore                          Algoritmo
  ------------- ------------------------------ ---------------------------- -------------------------------------- ----------------
  Squared error $(y_i - f(x_i))^2$             $y_i - f(x_i)$               $E[y \mid x_i]$                        L2Boosting

  Absolute      $\lvert y_i - f(x_i) \rvert$   $\text{sgn}(y_i - f(x_i))$   $\text{median}(y \mid x_i)$            Gradient
  error                                                                                                            boosting

  Exponential   $\exp(-y_i f(x_i))$            $-y_i \exp(-y_i f(x_i))$     $\frac{1}{2} \log \frac{\pi}{1-\pi}$   AdaBoost
  loss                                                                                                             

  Logloss       $\log(1 + e^{-y_i f(x_i)})$    $y_i - \pi_i$                $\frac{1}{2} \log \frac{\pi}{1-\pi}$   LogitBoost
  ---------------------------------------------------------------------------------------------------------------------------------

Per problemi di classificazione binaria, assumiamo $y_i \in \{-1, +1\}$,
$y_i \in \{0, 1\}$ e $\pi = \text{sigm}(2f(x_i))$. Per problemi di
regressione, assumiamo $y_i \in \mathbb{R}$.

### Forward Stagewise Additive Modeling

Poiché trovare l'$f$ ottimale è difficile, lo affrontiamo
**sequenzialmente**. Inizializziamo definendo:

$$f_0(x) = \arg\min_\theta \sum_{i=1}^{N} L(y_i, f(x_i; \theta))$$

-   Es. se usiamo squared error, possiamo impostare $f_0(x) = \bar{y}$
-   Se usiamo log-loss o exponential loss, possiamo impostare
    $f_0(x) = \frac{1}{2} \log \frac{\pi}{1-\pi}$, dove
    $\pi = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i = 1)$

Poi all'iterazione $m$ calcoliamo:

$$(\beta_m, \theta_m) = \arg\min_{(\beta, \theta)} \sum_{i=1}^{N} L(y_i, f_{m-1}(x_i) + \beta \phi(x_i; \theta))$$

e poi impostiamo:

$$f_m(x) = f_{m-1}(x) + \beta_m \phi(x; \theta_m)$$

> Il punto chiave è che **non torniamo indietro** per aggiustare i
> parametri precedenti. Questo è il motivo per cui il metodo è chiamato
> **forward stagewise additive modeling**.

------------------------------------------------------------------------

## 📉 L2 Boosting

Supponiamo di usare squared error loss. Allora al passo $m$ la loss ha
la forma:

$$L(y_i, f_{m-1}(x_i) + \beta \phi(x_i; \theta)) = (r_{im} - \phi(x_i; \theta))^2$$

dove $r_{im} = y_i - f_{m-1}(x_i)$ è il **residuo corrente**, e abbiamo
impostato $\beta = 1$ senza perdita di generalità.

> Quindi possiamo trovare la nuova basis function usando il weak learner
> per **predire $r_m$**. Questo è chiamato **L2boosting**, o **least
> squares boosting**.

------------------------------------------------------------------------

## 🎯 AdaBoost

Consideriamo un problema di classificazione binaria con **exponential
loss**. Al passo $m$ dobbiamo minimizzare:

$$L_m(\phi) = \sum_{i=1}^{N} \exp(-y_i(f_{m-1}(x_i) + \beta \phi(x_i))) = \sum_{i=1}^{N} w_{i,m} \exp(-\beta y_i \phi(x_i))$$

dove $w_{i,m} = \exp(-y_i f_{m-1}(x_i))$ è un peso applicato al caso
dati $i$, e $y_i \in \{-1, +1\}$.

Possiamo riscrivere questo obiettivo come:

$$L_m = (e^{\beta} - e^{-\beta}) \sum_{i=1}^{N} w_{i,m} \mathbb{I}(y_i \neq \phi(x_i)) + e^{-\beta} \sum_{i=1}^{N} w_{i,m}$$

La funzione ottimale può essere trovata applicando il weak learner a una
**versione pesata del dataset**, con pesi $w_{i,m}$.

### Algoritmo AdaBoost (M1)

    1. Inizializza i pesi: w_i = 1/N per i = 1, ..., N
    2. For m = 1 to M:
       a. Applica il weak learner ai dati pesati, ottieni φ_m
       b. Calcola l'errore pesato: err_m = Σ_i w_i · I(y_i ≠ φ_m(x_i)) / Σ_i w_i
       c. Calcola il peso del modello: β_m = log((1 - err_m) / err_m)
       d. Aggiorna i pesi: w_i ← w_i · exp(β_m · I(y_i ≠ φ_m(x_i)))
       e. Normalizza i pesi
    3. Output: f(x) = sign(Σ_m β_m φ_m(x))

-   Gli esempi misclassificati ricevono **più peso** nei round
    successivi
-   Il peso $\beta_m$ del modello è **maggiore** per i weak learner più
    accurati

------------------------------------------------------------------------

## 📊 LogitBoost

Il problema con l'exponential loss è che mette **molto peso sugli esempi
misclassificati**: - Questo rende il metodo molto sensibile agli
**outlier** (esempi con etichetta sbagliata) - Inoltre, $e^{-yf}$ non è
il logaritmo di alcuna pmf per variabili binarie $y \in \{-1, +1\}$; di
conseguenza **non possiamo recuperare stime di probabilità** da $f(x)$

Un'alternativa naturale è usare **logloss** invece: - Questo punisce gli
errori **solo linearmente** - Inoltre, significa che saremo in grado di
estrarre probabilità dalla funzione finale appresa, usando:

$$p(y = 1 \mid x) = \frac{e^{f(x)}}{e^{-f(x)} + e^{f(x)}} = \frac{1}{1 + e^{-2f(x)}}$$

L'obiettivo è minimizzare la log-loss attesa, data da:

$$L_m(\phi) = \sum_{i=1}^{N} \log(1 + \exp(-2y_i(f_{m-1}(x) + \phi(x_i))))$$

------------------------------------------------------------------------

## 📈 Boosting come Functional Gradient Descent

Piuttosto che derivare nuove versioni di boosting per ogni diversa
funzione di loss, è possibile derivare una versione **generica**, nota
come **gradient boosting**.

Per spiegare questo, immagina di minimizzare:

$$\bar{f} = \arg\min_f L(f)$$

dove $f = (f(x_1), \dots, f(x_N))$ sono i parametri. Risolveremo questo
stagewise, usando gradient descent. Al passo $m$, sia $g_m$ il gradiente
di $L(f)$ valutato a $f = f_{m-1}$:

$$g_{im} = \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}}$$

Poi facciamo l'aggiornamento:

$$f_m = f_{m-1} - \rho_m g_m$$

dove $\rho_m$ è la lunghezza del passo, scelta da:

$$\rho_m = \arg\min_\rho L(f_{m-1} - \rho g_m)$$

### Gradient Boosting

-   Se applichiamo questo algoritmo usando squared loss, recuperiamo
    **L2Boosting**
-   Se applichiamo questo algoritmo alla log-loss, otteniamo un
    algoritmo noto come **BinomialBoost**
-   Il vantaggio di questo rispetto a LogitBoost è che **non ha bisogno
    di essere in grado di fare weighted fitting**: applica semplicemente
    qualsiasi modello di regressione black-box al vettore del gradiente
-   Inoltre, è relativamente facile estenderlo al caso **multi-classe**

------------------------------------------------------------------------

## 🧩 Stacking

Un modo ovvio per stimare i pesi è usare:

$$w = \arg\min_w \sum_{i=1}^{N} L(y_i, \sum_{m=1}^{M} w_m f_m(x))$$

Tuttavia, questo risulterà in **overfitting**, con $w_m$ grande per il
modello più complesso. Una soluzione semplice è usare la
**cross-validation**. In particolare, possiamo usare la stima **LOOCV**:

$$w = \arg\min_w \sum_{i=1}^{N} L(y_i, \sum_{m=1}^{M} w_m f_m^{-i}(x))$$

dove $f_m^{-i}$ è il predittore ottenuto addestrando sui dati escludendo
$(x_i, y_i)$.

``` mermaid
flowchart TB
    A[Ensemble Learning] --> B[Bagging<br/>Random Forests]
    A --> C[Boosting<br/>sequenziale]
    A --> D[Stacking<br/>pesi via CV]
    C --> C1[L2Boosting]
    C --> C2[AdaBoost]
    C --> C3[LogitBoost]
    C --> C4[Gradient Boosting]
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 14_nlp_deep_learning.md ===== -->
```
# 10 --- Kernel e Support Vector Machines (SVM)

> *Riferimento: Murphy, MLPP Ch. 14*

------------------------------------------------------------------------

## 🧩 Introduzione

Finora abbiamo assunto che ogni esempio da classificare possa essere
rappresentato come un **feature vector a dimensione fissa**
$x_i \in \mathbb{R}^D$.

Tuttavia, per certi tipi di oggetti non è chiaro come rappresentarli al
meglio come feature vector a dimensione fissa: - Come rappresentiamo un
**documento di testo** o una **sequenza proteica**, che possono essere
di lunghezza variabile? - O un **albero evolutivo**, che ha dimensione e
forma variabile? - O una **struttura molecolare**, che ha complessa
geometria 3D?

**Un approccio:** assumere di avere un modo di misurare la
**similarità** tra oggetti, che non richiede di preprocessarli in
formato feature vector. Quando confrontiamo stringhe, possiamo calcolare
la **edit distance** tra loro.

Sia $\kappa(x, x') \geq 0$ una misura di similarità tra oggetti
$x, x' \in \mathcal{X}$, chiamata **kernel function**.

------------------------------------------------------------------------

## 📐 Funzione Kernel

Definiamo una **kernel function** come una funzione a valori reali di
due argomenti, $\kappa(x, x') \in \mathbb{R}$.

### RBF Kernels

-   La funzione è **simmetrica**, $\kappa(x, x') = \kappa(x', x)$ e
    **non negativa**
-   Il **squared exponential kernel** o **Gaussian kernel** è definito
    da:

$$\kappa(x, x') = \exp\left(-\frac{1}{2}(x - x')^T \Sigma^{-1}(x - x')\right)$$

-   Se $\Sigma$ è diagonale:

$$\kappa(x, x') = \exp\left(-\frac{1}{2}\sum_{j=1}^{D}\frac{1}{\sigma_j^2}(x_j - x_j')^2\right)$$

-   $\sigma_j$ può essere interpretato come la **lunghezza
    caratteristica** della dimensione $j$
-   Se $\sigma_j = \infty$ (ARD kernel), la dimensione corrispondente è
    **ignorata**
-   Se $\Sigma$ è sferica (**isotropic kernel**):

$$\kappa(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

-   Una matrice di covarianza $C$ è chiamata **isotropica** (o sferica)
    se è proporzionale alla matrice identità: $C = \lambda I$
-   Un esempio di **radial basis function (RBF) kernel**, poiché è solo
    una funzione di $\|x - x'\|$
-   $\sigma^2$ è nota come **bandwidth**

------------------------------------------------------------------------

## ✅ Mercer Kernels

Alcuni metodi richiedono che la kernel function soddisfi il requisito
che la **Gram matrix**, definita da:

$$K = \begin{pmatrix} \kappa(x_1, x_1) & \cdots & \kappa(x_1, x_N) \\ \vdots & \ddots & \vdots \\ \kappa(x_N, x_1) & \cdots & \kappa(x_N, x_N) \end{pmatrix}$$

sia **definita positiva** per qualsiasi insieme di input.

-   Chiamiamo tale kernel un **Mercer kernel**, o **positive definite
    kernel**
-   **Teorema di Mercer:** se la Gram matrix è definita positiva,
    possiamo calcolare una decomposizione in autovettori:

$$\zeta = U^T \Lambda U$$

dove $\Lambda$ è una matrice diagonale di autovalori $\lambda_i > 0$.

------------------------------------------------------------------------

## 📝 Esempio: Polynomial Kernel

Consideriamo il polynomial kernel
$\kappa(x, x') = (\gamma x^T x' + r)^M$ dove $r > 0$.

-   Il feature vector corrispondente $\phi(x)$ conterrà **tutti i
    termini fino al grado $M$**
-   Per $M = 2$, $\gamma = r = 1$ e $x, x' \in \mathbb{R}^2$:

$$(1 + x^T x')^2 = (1 + x_1 x_1' + x_2 x_2')^2 = 1 + 2x_1 x_1' + 2x_2 x_2' + (x_1 x_1')^2 + (x_2 x_2')^2 + 2x_1 x_1' x_2 x_2'$$

che può essere scritto come $\phi(x)^T \phi(x')$, dove:

$$\phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, x_2^2, \sqrt{2}x_1 x_2]^T$$

> Usare questo kernel è equivalente a lavorare in uno **spazio di
> feature a 6 dimensioni**.

### Linear Kernels

Il **linear kernel** è il caso più semplice:

$$\kappa(x, x') = x^T x'$$

-   Corrisponde a lavorare nello spazio di feature originale
-   Non aggiunge non linearità, ma permette di usare le tecniche
    kernelizzate su problemi lineari

------------------------------------------------------------------------

## 📄 Kernel per Confrontare Documenti

Quando si esegue classificazione o retrieval di documenti, è utile avere
un modo di confrontare due documenti $x_i$ e $x_j$.

-   Se usiamo una rappresentazione **bag of words**, dove $x_{ik}$ è il
    numero di volte che la parola $k$ occorre nel documento $i$,
    possiamo usare la **cosine similarity**:

$$\kappa(x_i, x_j) = \frac{x_i^T x_j}{\|x_i\|_2 \|x_j\|_2}$$

**Questo metodo semplice non funziona molto bene, per due motivi
principali:** 1. Se $x_i$ ha una parola in comune con $x_j$, è
considerato simile, anche se alcune parole popolari (come "the" o "and")
occorrono in molti documenti e quindi **non sono discriminative** 2. Se
una parola discriminativa occorre molte volte in un documento, la
similarità è **artificialmente aumentata**, anche se l'uso delle parole
tende a essere *bursty* (una volta che una parola è usata in un
documento è molto probabile che sia usata di nuovo)

------------------------------------------------------------------------

## 🤖 Kernel Machine

Definiamo una **kernel machine** come un modello lineare generativo
(GLM) dove il feature vector di input ha la forma di un **feature vector
kernelizzato**:

$$\phi(x) = [\kappa(x, \mu_1), \dots, \kappa(x, \mu_K)]$$

dove $\mu_k \in \mathcal{X}$ sono un insieme di $K$ centroidi.

-   Un approccio semplice è fare di ogni esempio $x_i$ un prototipo
-   Come scegliamo i centroidi $\mu_k$?

$$\phi(x) = [\kappa(x, x_1), \dots, \kappa(x, x_N)]$$

-   Vediamo $D = N$, quindi abbiamo **tanti parametri quanti punti
    dati**
-   Possiamo usare qualsiasi prior che promuove la sparsità per
    selezionare efficientemente un sottoinsieme degli esempi di
    training: **sparse vector machine**
-   $\ell_1$ o $\ell_2$ regularized vector machine
-   Un approccio molto popolare per creare una kernel machine sparsa è
    usare una **support vector machine (SVM)**

------------------------------------------------------------------------

## 🎩 The Kernel Trick

Piuttosto che definire il nostro feature vector in termini di kernel,
$\phi(x) = [\kappa(x, x_1), \dots, \kappa(x, x_N)]$, possiamo invece
**lavorare con i feature vector originali** $x$, ma **modificare
l'algoritmo** così che sostituisca tutti i prodotti interni della forma
$\langle x, x' \rangle$ con una chiamata alla kernel function
$\kappa(x, x')$.

``` mermaid
flowchart LR
    A[Feature vector originali x] --> B[Algoritmo]
    B --> C{Prodotto interno ⟨x, x'⟩}
    C --> D[Sostituisci con kernel κ(x, x')]
    D --> E[Lavora in spazio di feature implicito]
```

------------------------------------------------------------------------

## 🛠️ Support Vector Machines (SVM)

-   Un modo per derivare una kernel machine sparsa è usare un GLM con
    kernel basis functions, più un prior che promuove la sparsità come
    $\ell_1$
-   Un approccio alternativo è **cambiare la funzione obiettivo** da
    negative log likelihood a qualche altra loss function

Consideriamo la **ℓ2 regularized empirical risk function**:

$$J(w, \lambda) = \sum_{i=1}^{N} L(y_i, \hat{y}_i) + \lambda \|w\|^2$$

dove $\hat{y}_i = w^T x_i + w_0$: - Se $L$ è quadratic loss →
equivalente a **ridge regression** - Se $L$ è la log-loss → equivalente
a **logistic regression**

Nel caso della ridge regression, la soluzione ha la forma
$\hat{w} = (X^T X + \lambda I)^{-1} X^T y$, e le predizioni prendono la
forma $\hat{w}_0 + \hat{w}^T x$. Possiamo riscrivere queste equazioni in
un modo che coinvolge solo prodotti interni della forma $x^T x'$, che
possiamo sostituire con chiamate a una kernel function $\kappa(x, x')$.
Questo è **kernelizzato, ma non sparso**.

### Kernelized Ridge Regression

La **kernelized ridge regression** applica il kernel trick alla ridge
regression:

-   La soluzione può essere scritta come combinazione lineare dei dati
    di training: $\hat{w} = \sum_i \alpha_i x_i$
-   Le predizioni diventano:

$$\hat{y}(x) = \sum_{i=1}^{N} \alpha_i \kappa(x_i, x)$$

-   I coefficienti $\alpha$ si ottengono risolvendo un sistema lineare
    che coinvolge la Gram matrix $K$
-   **Svantaggio:** la soluzione dipende da **tutti** i punti di
    training (non sparsa)

### SVM /2

Se sostituiamo la quadratic/log loss con qualche altra loss function,
possiamo garantire che la soluzione sia **sparsa**, così che le
predizioni dipendano solo da un sottoinsieme dei dati di training, noti
come **support vectors**.

> Questa combinazione del **kernel trick** più una **loss function
> modificata** è nota come **support vector machine (SVM)**.

------------------------------------------------------------------------

## 📊 SVM per la Classificazione

Consideriamo la **hinge loss**:

$$L(y, \eta) = \max(0, 1 - y\eta) = (1 - y\eta)_+$$

dove $\eta = f(x) = w^T x + w_0$.

L'obiettivo complessivo ha la forma:

$$\min_{w, w_0} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} (1 - y_i f(x_i))_+$$

Introducendo le **slack variables** $\xi_i$ questo è equivalente a:

$$\min_{w, w_0, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} \xi_i, \quad \text{s.t. } \xi_i \geq 0, \quad y_i(x_i^T w + w_0) \geq 1 - \xi_i$$

Al tempo di test la predizione è fatta da:

$$\hat{y}(x) = \text{sgn}(f(x)) = \text{sgn}(\hat{w}^T x + \hat{w}_0) = \text{sgn}\left(\hat{w}_0 + \sum_{i=1}^{N} \alpha_i \kappa(x_i, x)\right)$$

------------------------------------------------------------------------

## 📈 SVM per la Regressione

Il problema con la kernelized ridge regression è che il vettore
soluzione $w$ dipende da **tutti** gli input di training. Cerchiamo ora
un metodo per produrre una stima **sparsa**.

### Epsilon-Insensitive Loss

$$L_\epsilon(y, \hat{y}) = \begin{cases} 0 & \text{se } |y - \hat{y}| < \epsilon \\ |y - \hat{y}| - \epsilon & \text{altrimenti} \end{cases}$$

-   Questo significa che qualsiasi punto che giace dentro un **ε-tube**
    attorno alla predizione **non è penalizzato**
-   La funzione obiettivo corrispondente è scritta come:

$$J = C \sum_{i=1}^{N} L_\epsilon(y_i, \hat{y}_i) + \frac{1}{2}\|w\|^2$$

dove $C = 1/\lambda$ è una costante di regolarizzazione.

``` mermaid
flowchart LR
    A[Kernel Trick] --> B[SVM]
    C[Loss modificata] --> B
    B --> D[Classificazione<br/>hinge loss]
    B --> E[Regressione<br/>ε-insensitive loss]
    D --> F[Support vectors]
    E --> F
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 11_deep_learning.md ===== -->
```

------------------------------------------------------------------------

# PARTE V --- Deep Learning e modelli generativi

# 11 --- Deep Learning

> *Riferimento: Goodfellow, Bengio, Courville, Deep Learning*

------------------------------------------------------------------------

## 🧠 Cos'è il Deep Learning

Il Deep Learning è un **sotto-campo del machine learning** che si occupa
di algoritmi ispirati alla struttura e alla funzione del cervello,
chiamati **reti neurali artificiali**.

``` mermaid
flowchart LR
    A[Deep Learning] --> B[Reti Neurali Artificiali]
    B --> C[Ispirate al cervello]
    C --> D[Apprendimento gerarchico di feature]
```

------------------------------------------------------------------------

## 🧬 Il Neurone

-   Il neurone è ottimizzato per **ricevere informazioni**
-   Ognuna di queste connessioni in arrivo è **dinamicamente rafforzata
    o indebolita** in base a quanto spesso viene usata
-   Il neurone riceve i suoi input lungo i **dendriti**
-   La forza di ogni connessione determina il contributo dell'input
    all'output del neurone
-   Dopo essere stati pesati dalla forza delle rispettive connessioni,
    gli input sono **sommati** nel corpo cellulare
-   Questa somma è poi trasformata in un nuovo segnale che viene
    propagato lungo l'**assone** della cellula e inviato ad altri
    neuroni

### Neurone Artificiale (1943)

Proprio come nei neuroni biologici, il neurone artificiale prende un
certo numero di input, $x = [x_1, x_2, \dots, x_n]$, ognuno dei quali è
moltiplicato per un peso specifico, $w = [w_1, w_2, \dots, w_n]$.

Gli input pesati sono sommati:

$$z = \sum_{i=0}^{n} w_i x_i$$

$z$ è poi passato attraverso una funzione $f$ per produrre l'output
$y = f(z) = f(xw + b)$, dove $b$ è il termine di bias.

------------------------------------------------------------------------

## 🏗️ Reti Neurali Feed-Forward

-   Un singolo neurone artificiale **non è abbastanza espressivo** per
    risolvere problemi di apprendimento complicati
-   I neuroni nel cervello umano sono organizzati in **strati**
-   L'informazione fluisce da uno strato all'altro finché l'input
    sensoriale è convertito in comprensione concettuale

``` mermaid
flowchart LR
    subgraph Input["Input Layer"]
        I1[x1]
        I2[x2]
        I3[x3]
    end
    subgraph Hidden["Hidden Layer"]
        H1[h1]
        H2[h2]
    end
    subgraph Output["Output Layer"]
        O1[y1]
        O2[y2]
    end
    I1 --> H1
    I1 --> H2
    I2 --> H1
    I2 --> H2
    I3 --> H1
    I3 --> H2
    H1 --> O1
    H1 --> O2
    H2 --> O1
    H2 --> O2
```

------------------------------------------------------------------------

## ⚡ Funzioni di Attivazione

  ----------------------------------------------------------------------------------------------------------
  Funzione                    Formula                                                 Range
  --------------------------- ------------------------------------------------------- ----------------------
  **Linear**                  $f(x) = cx$                                             $(-\infty, +\infty)$

  **Sigmoid**                 $f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$               $(0, 1)$

  **Tanh**                    $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$   $(-1, +1)$

  **ReLU**                    $f(x) = \max(0, x)$                                     $(-\infty, +\infty)$

  **Leaky ReLU**              $f(x) = \max(0, z_i) + 0.01 \min(0, z_i)$               $(-\infty, +\infty)$
  ----------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧩 Il Problema XOR

La funzione **XOR** è un'operazione su due input binari che restituisce
1 se gli input sono diversi, 0 altrimenti.

-   Il problema XOR è **non linearmente separabile**: nessuna singola
    retta può separare le classi
-   Un singolo neurone (perceptron) **non può** risolvere XOR
-   Serve una rete con **almeno un hidden layer** con non linearità

``` mermaid
flowchart LR
    A[x1] --> H1[hidden]
    B[x2] --> H1
    A --> H2[hidden]
    B --> H2
    H1 --> O[output]
    H2 --> O
```

-   Valutato su tutto il training set, un modello lineare fallisce su
    XOR
-   Introduciamo una rete feed-forward semplice con un hidden layer non
    lineare:
    -   $f_1(x) = W_1 x$ (primo strato)
    -   $f_2(h) = h w$ (secondo strato)
-   La non linearità nell'hidden layer permette di risolvere il problema

------------------------------------------------------------------------

## 📉 Gradient Descent

-   Il **gradient descent** è l'algoritmo di ottimizzazione più comune
    nel deep learning
-   È un algoritmo di ottimizzazione **del primo ordine** (considera
    solo la prima derivata quando esegue gli aggiornamenti sui
    parametri)
-   La dimensione del passo che facciamo a ogni iterazione per
    raggiungere il minimo locale è determinata dal **learning rate**
    $\alpha$
-   A ogni iterazione, aggiorniamo i parametri nella **direzione opposta
    del gradiente** della funzione obiettivo $J(w)$ rispetto ai
    parametri, dove il gradiente dà la direzione della salita più ripida
-   Quindi, seguiamo la direzione della pendenza in discesa finché
    raggiungiamo un minimo locale

$$w \leftarrow w - \alpha \nabla_w J(w)$$

### Varianti del Gradient Descent

  ------------------------------------------------------------------------
  Variante        Aggiornamento           Vantaggi        Svantaggi
  --------------- ----------------------- --------------- ----------------
  **Batch GD**    Su tutto il dataset     Convergenza     Costoso, può
                                          stabile         fermarsi in
                                                          minimi locali

  **Mini-Batch    Su un sottoinsieme      Più veloce del  Non converge in
  GD**            (mini-batch)            batch, meno     modo netto
                                          rumoroso dello  
                                          stocastico      

  **Stochastic    Su un singolo punto     Gestisce la     Molto rumoroso
  GD**                                    ridondanza, può 
                                          uscire da       
                                          minimi locali   
  ------------------------------------------------------------------------

-   Il **mini-batch gradient descent** somma su un numero inferiore di
    esempi
-   **Vantaggi:** più veloce della versione batch, seleziona casualmente
    i mini-batch
-   **Svantaggi:** non converge in modo netto

------------------------------------------------------------------------

## 🔄 Backpropagation

Supponiamo di avere una NN di classificazione con 1 input e 2 output, e
un hidden layer non lineare con 1 neurone:

$$y = \text{sigmoid}(f(g(x w_h) + b_0))$$

-   La funzione di attivazione non lineare usata nell'hidden layer di
    questo esempio è la **Gaussian radial basis function (RBF)**:

$$g(z_h) = e^{-z_h^2}$$

Ogni iterazione della backpropagation consiste in due passi:

1.  Un passo di **forward propagation** per calcolare l'output della
    rete
2.  Un passo di **backward propagation** in cui l'errore alla fine della
    rete è propagato all'indietro attraverso tutti i neuroni mentre
    aggiorna i loro parametri

``` mermaid
flowchart LR
    A[Forward propagation<br/>calcola output] --> B[Calcola errore]
    B --> C[Backward propagation<br/>propaga errore all'indietro]
    C --> D[Aggiorna parametri]
    D --> A
```

------------------------------------------------------------------------

## ⚠️ Il Problema del Vanishing Gradient

Man mano che più strati che usano certe funzioni di attivazione vengono
aggiunti alle reti neurali, i **gradienti della loss function si
avvicinano a zero**, rendendo la rete difficile da addestrare.

-   Il problema del vanishing gradient **dipende dalla scelta della
    funzione di attivazione**
-   Le funzioni di attivazione comuni "schiacciano" il loro input in un
    range di output molto piccolo in modo molto non lineare
-   La **sigmoid** mappa gli input su un range "piccolo" di $[0, 1]$
-   In queste regioni dello spazio di input, anche un grande cambiamento
    nell'input produrrà un piccolo cambiamento nell'output --- quindi il
    gradiente è piccolo
-   Ci sono grandi regioni dello spazio di input che sono mappate su un
    range estremamente piccolo

------------------------------------------------------------------------

## 💰 Funzioni di Costo

Un aspetto importante del design di una rete neurale profonda è la
**scelta della funzione di costo**.

-   Nella maggior parte dei casi, il nostro modello parametrico
    definisce una distribuzione $p(y \mid x; \theta)$ e usiamo
    semplicemente il principio di **maximum likelihood**
-   Usiamo la **cross-entropy** tra i dati di training e le predizioni
    del modello come funzione di costo
-   Un approccio più semplice è predire semplicemente qualche statistica
    di $y$ condizionata a $x$
-   La funzione di costo totale usata per addestrare una rete neurale
    spesso combina una delle funzioni di costo primarie con un **termine
    di regolarizzazione**

### Linee Guida per Output Layer e Loss Function

  -------------------------------------------------------------------------------
  Task                Output Layer                 Loss Function
  ------------------- ---------------------------- ------------------------------
  **Regressione**     1 nodo di output **lineare** **Mean Squared Error (MSE)**

  **Classificazione   1 nodo con attivazione       **Cross-Entropy** (Log loss)
  binaria**           **Sigmoid**                  

  **Classificazione   1 nodo per classe con        **Cross-Entropy**
  multi-classe**      attivazione **Softmax**      
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## ⚙️ Ottimizzatori

Oltre al gradient descent base, esistono ottimizzatori più avanzati:

  -----------------------------------------------------------------------
  Ottimizzatore                          Descrizione
  -------------------------------------- --------------------------------
  **Batch GD**                           Aggiorna su tutto il dataset

  **Mini-Batch SGD**                     Aggiorna su un sottoinsieme

  **Stochastic GD**                      Aggiorna su un singolo punto

  **Adam**                               Adatta il learning rate per ogni
                                         parametro usando momenti del
                                         gradiente

  **RMSProp**                            Adatta il learning rate
                                         dividendo per la radice della
                                         media dei quadrati dei gradienti
  -----------------------------------------------------------------------

-   **Adam** e **RMSProp** sono ottimizzatori con **adaptive learning
    rates**
-   Spesso convergono più velocemente e sono più robusti alla scelta del
    learning rate

------------------------------------------------------------------------

## 🛡️ Overfitting e Regolarizzazione

Un problema principale in ML è come fare un algoritmo che performi bene
**non solo sui dati di training, ma anche su nuovi input**.

-   Le strategie progettate per ridurre l'errore di test sono note
    collettivamente come **regolarizzazione**
-   La regolarizzazione di uno stimatore funziona **scambiando bias
    aumentato per varianza ridotta**

Una famiglia di modelli in training: - **Esclude** il vero processo
generatore di dati → **underfitting** - **Corrisponde** al vero processo
generatore di dati - **Include** il processo generatore ma anche molti
altri possibili processi generatori → regime di **overfitting**

> L'obiettivo della regolarizzazione è portare un modello dal terzo
> regime al secondo regime.

### Dropout

Il **dropout** fornisce un metodo computazionalmente economico ma
potente per regolarizzare un'ampia famiglia di modelli.

-   Il dropout può essere pensato come un metodo per rendere il
    **bagging** praticabile per ensemble di molte grandi reti neurali
-   Il bagging coinvolge l'addestramento di più modelli e la valutazione
    di più modelli su ogni esempio --- **impraticabile** quando ogni
    modello è una grande rete neurale
-   Il dropout fornisce un'approssimazione economica all'addestramento e
    alla valutazione di un ensemble bagged di **esponenzialmente molte**
    reti neurali
-   Il dropout addestra l'ensemble consistente di tutti i sotto-reti che
    possono essere formate **rimuovendo unità non-output** da una rete
    base sottostante
-   Possiamo effettivamente rimuovere un'unità da una rete
    **moltiplicando il suo valore di output per zero**
-   Il dropout previene la **co-adaptation delle feature** (i nodi non
    possono fare affidamento su altri nodi specifici)

### Batch Normalization

La **batch normalization** normalizza gli input di ogni layer usando le
statistiche del batch:

1.  Calcola media $\mu_B$ e varianza $\sigma_B^2$ del batch
2.  Normalizza:
    $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
3.  Scala e trasla con parametri appresi $\gamma$ e $\beta$:
    $y = \gamma \hat{x} + \beta$

-   Riduce la **covariate shift** interna
-   Permette l'uso di learning rate più alti
-   Ha un effetto regolarizzante

### Early Stopping e Learning Rate Decay

-   **Early stopping:** ferma l'addestramento quando l'errore di
    validazione smette di migliorare
-   **Learning rate decay:** riduce gradualmente il learning rate
    durante l'addestramento per convergere più finemente

------------------------------------------------------------------------

## 🔄 Paradigmi Avanzati

### Multi-Task Learning

Il **Multi-Task Learning** condivide i **parametri generici nei layer
inferiori** per svolgere **più compiti simultaneamente**: - I layer
inferiori apprendono rappresentazioni condivise - I layer superiori sono
specifici per ogni task - Migliora la generalizzazione grazie alla
condivisione di conoscenza

### Semi-Supervised Learning

Il **Semi-Supervised Learning** condivide i parametri tra modelli
**generativi** $P(x)$ o $P(x, y)$ e **discriminativi** $P(y \mid x)$: -
Il modello generativo usa i dati non etichettati per apprendere la
struttura - Il modello discriminativo usa i dati etichettati per la
classificazione - La condivisione dei parametri permette di sfruttare
entrambi

------------------------------------------------------------------------

## 🔍 Reti Convoluzionali (CNN)

### L'Operazione di Convoluzione

La convoluzione è un'operazione su due funzioni, $x(t)$ e $w(t)$, di un
argomento a valori reali:

$$s(t) = \int x(a) w(t - a) da = (x * w)(t)$$

Nella terminologia CNN, il primo argomento della convoluzione è spesso
chiamato **input**, il secondo **kernel**. L'output è a volte chiamato
**feature map**.

Assumendo $x$ e $w$ definiti su $t$ intero, la convoluzione discreta è:

$$s(t) = (x * w)(t) = \sum_{a=-\infty}^{+\infty} x(a) w(t - a)$$

### Tre Idee Importanti

La convoluzione sfrutta tre idee importanti: - **Sparse interactions**
(interazioni sparse) - **Parameter sharing** (condivisione dei
parametri) - **Equivariant representations** (rappresentazioni
equivarianti)

### Pooling

Un tipico strato di una rete convoluzionale consiste di tre stadi: 1. Lo
strato esegue diverse convoluzioni in parallelo per produrre un insieme
di attivazioni lineari 2. Ogni attivazione lineare è passata attraverso
una funzione di attivazione non lineare 3. Usiamo una **pooling
function** per modificare ulteriormente l'output dello strato

Una pooling function sostituisce l'output della rete in una certa
posizione con una **statistica riassuntiva** degli output vicini: -
**Max-pooling:** riporta l'output massimo entro un vicinato
rettangolare - **Average-pooling:** riporta la media

------------------------------------------------------------------------

## 🔁 Reti Ricorrenti (RNN)

Le **recurrent neural networks (RNN)** sono una famiglia di reti neurali
per **elaborare dati sequenziali**.

-   Come una CNN è specializzata per elaborare una griglia di valori,
    una RNN è specializzata per elaborare una **sequenza di valori**
-   **Condivisione dei parametri** attraverso diverse parti del modello
-   La condivisione dei parametri rende possibile estendere e applicare
    il modello a esempi di forme diverse
-   Parametri separati per ogni valore dell'indice temporale non
    generalizzano a diverse lunghezze di sequenza
-   Es. risolvere il task di estrarre l'anno da "I went to Nepal in
    2009" e "In 2009, I went to Nepal"
-   Una RNN **condivide gli stessi pesi** attraverso diversi passi
    temporali

### LSTM

I modelli di sequenza più efficaci usati nelle applicazioni pratiche
sono chiamati **gated RNNs**: long short-term memory e reti basate sulla
gated recurrent unit.

-   Una **LSTM** è un'architettura RNN speciale più adatta a catturare
    **dipendenze a lungo termine** rispetto alle RNN vanilla
-   L'idea centrale nella LSTM è introdurre uno **stato di cella**
    $C_t$, più complicato della cella di memoria $s_t$ nelle RNN
    vanilla, dove l'informazione è aggiunta o rimossa da **strutture di
    gate**, composte da uno strato di rete neurale sigmoid e
    un'operazione di moltiplicazione

------------------------------------------------------------------------

## 🎨 Autoencoder

Gli autoencoder standard imparano a generare **rappresentazioni
compatte** e a ricostruire bene i loro input, ma a parte poche
applicazioni (come i denoising autoencoder), sono abbastanza limitati.

-   Il problema fondamentale con gli autoencoder, per la generazione, è
    che lo **spazio latente** in cui convertono i loro input e dove
    giacciono i loro vettori codificati, potrebbe **non essere
    continuo**, o non permettere una facile interpolazione
-   Quando costruisci un modello generativo, non vuoi prepararti a
    replicare la stessa immagine che metti dentro
-   Vuoi **campionare casualmente dallo spazio latente**, o generare
    variazioni su un'immagine di input, da uno spazio latente continuo

### Variational Autoencoders (VAE)

I VAE risolvono il problema della discontinuità dello spazio latente
imponendo una struttura probabilistica sullo spazio latente, permettendo
di campionare e interpolare in modo continuo.

#### Reparametrization Trick

Il **Reparametrization Trick** permette la **stochastic
back-propagation** attraverso il campionamento: - Invece di campionare
direttamente $z \sim q(z \mid x)$, si campiona un rumore
$\epsilon \sim \mathcal{N}(0, I)$ e si calcola
$z = \mu + \sigma \odot \epsilon$ - Questo rende il campionamento
**differenziabile** rispetto ai parametri - La perturbazione è espressa
come $f(x, z)$

$$\text{encoder: } z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### NADE (Neural Autoregressive Density Estimator)

Il **NADE** è un modello autoregressivo neurale per la **density
estimation**: - Fattorizza la distribuzione congiunta come prodotto di
condizionali: $p(x) = \prod_i p(x_i \mid x_{<i})$ - Usa uno schema di
**condivisione dei parametri** tra i nodi nascosti di gruppi
differenti - La sua propagazione in avanti ricorda le operazioni
dell'inferenza **mean-field** per completare dati mancanti nelle RBM

------------------------------------------------------------------------

## 🎲 Modelli Generativi

### Boltzmann Machines

Le Boltzmann machines sono state originariamente introdotte come un
approccio generale "connectionist" per apprendere **distribuzioni di
probabilità arbitrarie** su vettori binari.

-   La Boltzmann machine è un **modello basato sull'energia**, cioè
    definiamo la distribuzione di probabilità congiunta usando una
    funzione di energia:

$$P(\mathbf{x}) = \frac{\exp(-E(\mathbf{x}))}{Z}$$

-   $E(x)$ è la funzione di energia
-   $Z$ è la partition function, $Z = \sum_x \exp(-E(x))$
-   La funzione di energia della Boltzmann machine è data da:

$$E(x) = -x^T U x - b^T x$$

dove $U$ è la matrice dei pesi dei parametri del modello e $b$ è il
vettore dei parametri di bias.

### Restricted Boltzmann Machines (RBM)

Le RBM sono modelli grafici probabilistici **non diretti** contenenti
uno strato di variabili osservabili e un singolo strato di variabili
latenti.

-   Le RBM **rimuovono** le connessioni visible-visible e hidden-hidden
-   La funzione di energia della Boltzmann Machine è data da:

$$E(x, h) = -h^T W x - b^T x - c^T h$$

A causa della struttura specifica delle RBM, le unità visibili e
nascoste sono **condizionalmente indipendenti** date le altre:

$$p(h \mid x) = \prod_i p(h_i \mid x)$$

$$p(x \mid h) = \prod_i p(x_i \mid h)$$

### Generative Adversarial Networks (GAN)

Il modo più semplice per formulare l'apprendimento nelle GAN è come un
**gioco a somma zero**, in cui una funzione $v(g, d)$ determina il
payoff del discriminatore. Il generatore riceve $-v(g, d)$ come suo
payoff.

-   Durante l'apprendimento, ogni giocatore tenta di massimizzare il suo
    payoff, così che alla convergenza:

$$g^* = \arg_g \min_d \max_v v(g, d)$$

-   La scelta di default per $v$ è:

$$v(g, d) = \mathbb{E}_{x \sim data} \log d(x) + \mathbb{E}_{x \sim model} \log(1 - d(x))$$

Questo spinge il discriminatore a tentare di imparare a **classificare
correttamente i campioni come reali o falsi**. Simultaneamente, il
generatore tenta di **ingannare il classificatore** facendogli credere
che i suoi campioni siano reali.

-   Alla convergenza, i campioni del generatore sono **indistinguibili
    dai dati reali**, e il discriminatore produce 1/2 ovunque
-   Il discriminatore può poi essere scartato

``` mermaid
flowchart LR
    subgraph G["Generatore"]
        G1[Campioni falsi]
    end
    subgraph D["Discriminatore"]
        D1[Classifica reale/falso]
    end
    Z[Rumore casuale] --> G
    G --> D
    X[Dati reali] --> D
    D -->|Feedback| G
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 12_modelli_trattabili.md ===== -->
```
# 12 --- Tractable Probabilistic Models (TPM)

> *Riferimento: Vergari, Di Mauro, Van den Broeck --- Probabilistic
> Circuits*

------------------------------------------------------------------------

## 🧩 Introduzione

I **Tractable Probabilistic Models** (TPM) sono modelli probabilistici
che garantiscono **inferenza trattabile** (calcolabile in tempo
polinomiale) per classi di query come marginali, MAP, ecc.

``` mermaid
flowchart LR
    A[Modelli Logici e Probabilistici] --> B[Tractable e Intractable]
    B --> C[Modelli espressivi senza compromessi]
```

### Espressività vs Trattabilità

-   I modelli **completamente fattorizzati** sono molto trattabili ma
    **non espressivi**
-   I modelli **espressivi** non sono molto trattabili
-   I modelli **trattabili** non sono molto espressivi
-   I **probabilistic circuits** sono al "sweet spot" (punto ottimale)

------------------------------------------------------------------------

## 📦 Modelli Completamente Fattorizzati

Un grafo completamente disconnesso. Esempio: **Product of Bernoullis
(PoBs)**

$$p(X) = \prod_{i=1}^{n} p(x_i \mid \text{Pa}_{x_i})$$

-   Evidenza completa, marginali e inferenza MAP, MMAP è **lineare**!
-   Ma **decisamente non espressivo**...

------------------------------------------------------------------------

## 🔢 Fattorizzazioni sono Prodotti

**Divide and conquer complexity:**

$$p(X_1, X_2, X_3) = p(X_1) \cdot p(X_2) \cdot p(X_3)$$

Es. modellare una gaussiana multivariata con matrice di covarianza
diagonale.

------------------------------------------------------------------------

## ➕ Le Miscele sono Somme

Anche i modelli di miscela possono essere trattati come una semplice
unità computazionale su distribuzioni:

$$p(X) = w_1 \cdot p_1(X) + w_2 \cdot p_2(X)$$

------------------------------------------------------------------------

## 🧮 Probabilistic Circuits

Un **framework unificato** per modelli trattabili.

### I Vantaggi di Essere un Computational Graph

-   I calcoli che sono ripetuti possono essere **cached**!
    -   Amortizzare l'inferenza; condivisione di parametri/struttura
-   **Semantica operazionale chiara!**
-   **Differenziabile!**
    -   Ottimizzazione basata sul gradiente
-   **Trattabilità in termini di dimensione del circuito**
-   Le proprietà strutturali sul computational graph si mappano
    **pulitamente** su classi di query trattabili

------------------------------------------------------------------------

## ✅ Come Garantiamo la Trattabilità?

### Decomposability

Un **product node** è *decomposable* se i suoi figli dipendono da
**insiemi disgiunti di variabili** --- proprio come nella
fattorizzazione!

### Smoothness (aka completeness)

Un **sum node** è *smooth* se i suoi figli dipendono dagli **stessi
insiemi di variabili** --- altrimenti non tiene conto di alcune
variabili.

-   La smoothness può essere facilmente imposta \[Shih et al. 2019\]

### Determinism (aka selectivity)

Un **sum node** è *deterministic* se l'output di **solo un figlio** è
non zero per qualsiasi input --- es. se le loro distribuzioni hanno
**supporto disgiunto**.

``` mermaid
flowchart TB
    subgraph Decomp["Decomposability"]
        D1[Product node<br/>figli su variabili disgiunte]
    end
    subgraph Smooth["Smoothness"]
        S1[Sum node<br/>figli sugli stessi insiemi]
    end
    subgraph Det["Determinism"]
        T1[Sum node<br/>solo un figlio non-zero]
    end
    Decomp --> MAR[Trattabilità MAR/CON]
    Smooth --> MAR
    Det --> MAP[Trattabilità MAP]
```

------------------------------------------------------------------------

## 📊 Query Trattabili

-   **Tractable MAR/CON:** marginali e condizionali trattabili grazie a
    decomposability e smoothness
-   **Tractable MAP:** inferenza MAP trattabile grazie all'aggiunta del
    determinism

### Tipi di Query

  -----------------------------------------------------------------------
  Query                    Descrizione
  ------------------------ ----------------------------------------------
  **EVI** (Complete        Probabilità di un'assegnazione completa delle
  evidence)                variabili

  **MAR** (Marginal)       Probabilità marginale su un sottoinsieme di
                           variabili

  **CON** (Conditional)    Probabilità condizionale $P(Y \mid E = e)$

  **MAP** (Maximum A       Assegnazione più probabile di un sottoinsieme
  Posteriori)              di variabili

  **MMAP** (Marginal MAP)  MAP su un sottoinsieme, marginalizzando le
                           altre
  -----------------------------------------------------------------------

-   **EVI** è sempre trattabile (basta valutare il circuito)
-   **MAR/CON** richiedono decomposability e smoothness
-   **MAP** richiede anche determinism
-   **MMAP** è generalmente più difficile

------------------------------------------------------------------------

## 🏗️ Tipi di Probabilistic Circuits

### Arithmetic Circuits (ACs)

-   I parametri sono attaccati alle **foglie**
-   ...ma possono essere spostati sui bordi dei sum node \[Rooshenas et
    al. 2014\]
-   Vedi anche i relativi AND/OR search spaces \[Dechter et al. 2007\]

### Sum-Product Networks (SPNs)

-   Gli SPN deterministici sono anche chiamati **selective** \[Peharz et
    al. 2014a\]

### Cutset Networks (CNets)

Un CNet \[Rahman et al. 2014\] è un **weighted model-tree** \[Dechter et
al. 2007\] le cui foglie sono **tree Bayesian networks**.

-   Possono essere rappresentati come probabilistic circuits

### Probabilistic Sentential Decision Diagrams (PSDD)

Un'altra famiglia di probabilistic circuits basata sui sentential
decision diagrams.

------------------------------------------------------------------------

## 📈 Apprendimento dei Parametri

La distribuzione di un sum node $p(X)$ può essere interpretata come una
**distribuzione marginale** di $p(X, Z)$ su $X$ e una variabile latente
$Z$:

-   $p(X \mid Z = k)$
-   $p(Z = k) = w_k$

Anche le distribuzioni delle foglie potrebbero essere parametrizzate da
$\theta$.

L'apprendimento dei parametri coinvolge l'apprendimento sia dei
parametri dei sum node che delle foglie $(w, \theta)$.

------------------------------------------------------------------------

## 🏗️ Structure Learning

### LearnSPN

Apprendere sia la struttura che i parametri di un circuito partendo da
una **data matrix**.

-   Cercare **sotto-popolazioni** nei dati (clustering) per introdurre
    sum node...
-   Gens et al., "Learning the Structure of Sum-Product Networks", 2013

------------------------------------------------------------------------

## 🎲 Ensemble di Probabilistic Circuits

Per mitigare problemi come la scarsa accuratezza di un singolo modello e
la loro tendenza all'overfitting, i circuiti possono essere impiegati
come componenti di una **miscela**:

$$p(X) = \sum_{i=1}^{K} \lambda_i \mathcal{C}_i(X), \quad \lambda_i \geq 0, \quad \sum_{i=1}^{K} \lambda_i = 1$$

-   Impiegare EM per apprendere alternativamente sia i pesi che i
    componenti della miscela
-   Problemi di convergenza e instabilità di EM → **impraticabile**

### Bagging Probabilistic Circuits

-   Più efficiente di EM
-   I coefficienti della miscela sono impostati **ugualmente probabili**
-   I componenti della miscela possono essere appresi
    **indipendentemente** su diversi bootstrap
-   Aggiungere proiezione di random subspace alle reti bagged (come per
    i CNets) → più efficiente del bagging

### Boosting Probabilistic Circuits

**BDE (boosting density estimation):** cresce sequenzialmente
l'ensemble, aggiungendo un weak base learner a ogni stadio.

-   A ogni passo di boosting $m$, trova un weak learner $c_m$ e un
    coefficiente $\eta_m$ che massimizzano la weighted LL del nuovo
    modello:

$$f_m = (1 - \eta_m) f_{m-1} + \eta_m c_m$$

**GBDE:** una generalizzazione basata su kernel di BDE --- algoritmo in
stile AdaBoost con **sequential EM**: - A ogni passo $m$, ottimizza
congiuntamente $\eta_m$ e $c_m$ mantenendo $f_{m-1}$ fisso

------------------------------------------------------------------------

## 🔄 Online Learning

L'apprendimento online permette di aggiornare i parametri del circuito
**incrementally** man mano che arrivano nuovi dati, senza dover
riaddestrare da zero.

------------------------------------------------------------------------

## 🧠 Knowledge Compilation

La **knowledge compilation** è il processo di trasformare una
rappresentazione logica/probabilistica in una forma che rende
l'inferenza trattabile (es. trasformare un albero di decisione in un
circuito).

------------------------------------------------------------------------

## 🖼️ Applicazioni

### Computer Vision

-   **BACK-MPE:** ricostruzione e inpainting di immagini
-   **BACK-ORIG:** esempi di ricostruzioni di immagini di volti (metà
    coperta)
-   Poon et al., "Sum-Product Networks: a New Deep Architecture", 2011
-   Sguerra et al., "Image classification using sum-product networks for
    autonomous flight of micro aerial vehicles", 2016

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 13_boosting_ensemble.md ===== -->
```

------------------------------------------------------------------------

# PARTE VI --- Natural Language Processing

# 14 --- Natural Language Processing con Deep Learning

> *Riferimento: Christopher Manning, CS224N Stanford*

------------------------------------------------------------------------

## 🧠 Introduzione

Il Natural Language Processing (NLP) con Deep Learning combina tecniche
di reti neurali per elaborare e comprendere il linguaggio naturale.

------------------------------------------------------------------------

## 📖 Come Rappresentiamo il Significato di una Parola?

### Rappresentare le Parole come Simboli Discreti

Nel NLP tradizionale, consideriamo le parole come **simboli discreti**
(hotel, conference, motel) --- una **rappresentazione localista**.

Tali simboli per le parole possono essere rappresentati da **one-hot
vectors**:

    motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
    hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]

-   **Dimensione del vettore** = numero di parole nel vocabolario (es.
    500,000)

### Rappresentare le Parole per il loro Contesto

**Semantica distribuzionale:** il significato di una parola è dato dalle
parole che appaiono frequentemente vicino.

> *"You shall know a word by the company it keeps"* (J. R. Firth 1957)

-   Una delle idee di maggior successo del moderno NLP statistico!
-   Quando una parola $w$ appare in un testo, il suo contesto è
    l'insieme di parole che appaiono vicino (entro una finestra di
    dimensione fissa)
-   Usa i molti contesti di $w$ per costruire una rappresentazione di
    $w$

### Word Vectors

Costruiamo un **vettore denso** per ogni parola, scelto così che sia
simile ai vettori di parole che appaiono in contesti simili.

-   I word vectors sono anche chiamati **word embeddings** o (neural)
    word representations
-   Sono una **distributed representation**

------------------------------------------------------------------------

## 🎯 Word2vec

**Word2vec** (Mikolov et al. 2013) è un framework per apprendere word
vectors.

### Funzione Obiettivo

Per ogni posizione $t = 1, \dots, T$, predici le parole di contesto
entro una finestra di dimensione fissa $m$, data la parola centrale
$w_t$.

**Data likelihood:**

$$L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(w_{t+j} \mid w_t; \theta)$$

**Funzione obiettivo** $J(\theta)$ è la (media) negative log likelihood:

$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t; \theta)$$

> Minimizzare la funzione obiettivo ⟺ Massimizzare l'accuratezza
> predittiva

### Funzione di Predizione

$$P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

-   ① **Dot product** confronta la similarità di $o$ e $c$ --- dot
    product più grande = probabilità più grande
-   ② **Esponenziale** rende tutto positivo
-   ③ **Normalizza** sull'intero vocabolario per dare una distribuzione
    di probabilità

Questo è un esempio della funzione **softmax**:

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)} = p_i$$

-   La softmax mappa valori arbitrari $x_i$ a una distribuzione di
    probabilità $p_i$
-   "max" perché amplifica la probabilità del più grande $x_i$
-   "soft" perché assegna ancora qualche probabilità ai più piccoli
    $x_i$
-   Frequentemente usata nel Deep Learning

### Due Varianti del Modello

1.  **Skip-grams (SG):** predici le parole di contesto ("outside") data
    la parola centrale
2.  **Continuous Bag of Words (CBOW):** predici la parola centrale da
    (un bag di) parole di contesto

### Efficienza nell'Addestramento

Il calcolo della softmax naïve sull'intero vocabolario è **costoso** (il
vocabolario può avere centinaia di migliaia di parole). Due tecniche
principali per l'efficienza:

1.  **Negative Sampling:** invece di normalizzare su tutto il
    vocabolario, campiona un piccolo numero di parole "negative" (non di
    contesto) e ottimizza per distinguere le parole di contesto reali da
    quelle campionate
2.  **Hierarchical Softmax:** usa un albero binario sul vocabolario per
    ridurre il costo della normalizzazione da $O(|V|)$ a $O(\log |V|)$

### Decomposizione SVD

Un approccio classico per ottenere word vectors dai **co-occurrence
counts**: - Costruisci la co-occurrence matrix $X$ - Applica la
**decomposizione SVD** (Singular Value Decomposition) per ridurre la
dimensionalità - I vettori risultanti catturano la struttura semantica
dei dati

------------------------------------------------------------------------

## 📉 Ottimizzazione: Gradient Descent

-   Abbiamo una funzione di costo $J$ che vogliamo minimizzare
-   **Gradient Descent** è un algoritmo per minimizzare $J$
-   **Idea:** per il valore corrente di $J$, calcola il gradiente di
    $J$, poi fai un piccolo passo nella direzione del gradiente
    negativo. Ripeti.

### Stochastic Gradient Descent (SGD)

**Problema:** $J$ è una funzione di tutte le finestre nel corpus
(potenzialmente miliardi!)

-   Quindi $\nabla_\theta J(\theta)$ è molto costoso da calcolare
-   Aspetteresti molto tempo prima di fare un singolo aggiornamento!
-   Pessima idea per quasi tutte le reti neurali!

**Soluzione:** Stochastic gradient descent (SGD) - Campiona
ripetutamente finestre, e aggiorna dopo ognuna

    while True:
        window = Sample_window(corpus)
        theta_grad = evaluate_gradient(J, window, theta)
        theta = theta - alpha * theta_grad

------------------------------------------------------------------------

## 📊 Perché non Catturare i Co-occurrence Counts Direttamente?

Costruire una **co-occurrence matrix** $X$: - **2 opzioni:** finestre vs
documento completo - **Window:** simile a word2vec, usa una finestra
attorno a ogni parola → cattura qualche informazione sintattica e
semantica - La **word-document co-occurrence matrix** darà topic
generali (tutti i termini sportivi avranno voci simili) portando alla
"Latent Semantic Analysis"

### Count-based vs Direct Prediction

  -----------------------------------------------------------------------
  Count-based (LSA, HAL, COALS)     Direct prediction (Skip-gram/CBOW)
  --------------------------------- -------------------------------------
  Training veloce                   Scala con la dimensione del corpus

  Uso efficiente delle statistiche  Uso inefficiente delle statistiche

  Usato principalmente per          Genera performance migliorata su
  catturare similarità di parole    altri task

  Importanza sproporzionata data ai Può catturare pattern complessi oltre
  grandi counts                     la similarità di parole
  -----------------------------------------------------------------------

### GloVe: Combinare il Meglio di Entrambi i Mondi

**GloVe** \[Pennington, Socher, and Manning, EMNLP 2014\]:

$$w_i \cdot w_j = \log P(i \mid j)$$

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

-   Training veloce
-   Scalabile a corpora enormi
-   Buona performance anche con corpus piccoli e vettori piccoli

------------------------------------------------------------------------

## 🔤 Word Senses e Ambiguità

La maggior parte delle parole ha **molti significati**! - Soprattutto le
parole comuni - Soprattutto le parole che esistono da molto tempo -
Esempio: *pike* - Un singolo vettore cattura tutti questi significati o
abbiamo un pasticcio?

------------------------------------------------------------------------

## 🏷️ Classificazione: Revisione e Notazione

Generalmente, abbiamo un training dataset consistente di campioni:

$$\{x_i, y_i\}_{i=1}^{N}$$

-   $x_i$ sono input, es. parole (indici o vettori!), frasi, documenti,
    ecc. (dimensione $d$)
-   $y_i$ sono etichette (una delle $C$ classi) che cerchiamo di
    predire, per esempio:
    -   Classi: sentiment (+/-), named entities, decisione buy/sell
    -   Altre parole
    -   Più tardi: sequenze multi-parola

### Softmax Classifier

Un **softmax classifier** è un classificatore lineare che usa la
funzione softmax per produrre una distribuzione di probabilità sulle
classi:

$$p(y = c \mid x) = \frac{\exp(w_c^T x)}{\sum_{c'=1}^{C} \exp(w_{c'}^T x)}$$

-   Ogni classe $c$ ha il suo vettore di pesi $w_c$
-   L'addestramento usa la **cross-entropy loss**:

$$L = -\log p(y \mid x)$$

-   Questo è anche chiamato **logistic regression multiclasse**

------------------------------------------------------------------------

## 📚 Language Modeling

Il **Language Modeling** è il task di predire quale parola viene dopo:

> the students opened their \_\_\_\_\_\_

Più formalmente: data una sequenza di parole, calcola la distribuzione
di probabilità della parola successiva:

$$P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)})$$

dove $x^{(t+1)}$ può essere qualsiasi parola nel vocabolario
$V = \{w_1, \dots, w_{|V|}\}$.

Un sistema che fa questo è chiamato **Language Model**.

### n-gram Language Models

**Risposta (pre-Deep Learning):** apprendi un **n-gram Language Model**!

-   **Definizione:** un n-gram è un blocco di $n$ parole consecutive
    -   unigrams: 'the', 'students', 'opened', 'their'
    -   bigrams: 'the students', 'students opened', 'opened their'
    -   trigrams: 'the students opened', 'students opened their'
    -   4-grams: 'the students opened their'
-   **Idea:** raccogli statistiche su quanto frequenti sono i diversi
    n-gram e usali per predire la parola successiva

### Problemi di Sparsità con gli n-gram

Gli n-gram Language Models soffrono di **problemi di sparsità**:

-   **Problema 1:** se un n-gram non è mai stato visto nel corpus, la
    sua probabilità è 0 → la probabilità dell'intera frase diventa 0
-   **Problema 2:** anche con un corpus grande, la maggior parte degli
    n-gram per $n$ grande non appare mai
-   **Problema di storage:** il numero di possibili n-gram cresce
    esponenzialmente con $n$

**Soluzioni classiche:** - **Smoothing** (es. Laplace): assegnare
probabilità piccole ma non zero agli n-gram non visti - **Backoff:**
usare n-gram più corti quando quelli più lunghi non sono disponibili

### Valutare i Language Models

La metrica di valutazione standard per i Language Models è la
**perplexity**:

$$\text{perplexity} = \prod_{t=1}^{T} \left(\frac{1}{P_{LM}(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)})}\right)^{1/T}$$

Questo è uguale all'esponenziale della cross-entropy loss:

$$= \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log \hat{y}_{x_{t+1}}^{(t)}\right) = \exp(J(\theta))$$

> **Perplexity più bassa è meglio!**

------------------------------------------------------------------------

## 🔁 Recurrent Neural Networks (RNN)

Una famiglia di architetture neurali.

**Core idea:** applica gli stessi pesi $W$ ripetutamente.

### Un Semplice RNN Language Model

-   L'input sequenza può essere molto più lungo ora!
-   Le RNN condividono gli stessi pesi attraverso i passi temporali

### Problemi con Vanishing e Exploding Gradients

Le RNN soffrono di problemi di **vanishing** e **exploding gradients**
quando la sequenza è lunga.

### Long Short-Term Memory (LSTM)

Un tipo di RNN proposto da **Hochreiter e Schmidhuber nel 1997** come
soluzione al problema del vanishing gradients.

-   Al passo $t$, c'è uno **hidden state** e un **cell state** ---
    entrambi vettori di lunghezza $n$
-   La cella memorizza **informazione a lungo termine**
-   La LSTM può **leggere, cancellare e scrivere** informazione dalla
    cella
-   La cella diventa concettualmente simile alla RAM di un computer
-   La selezione di quale informazione è cancellata/scritta/letta è
    controllata da **tre gate corrispondenti**
-   I gate sono anche vettori di lunghezza $n$
-   A ogni timestep, ogni elemento dei gate può essere aperto (1),
    chiuso (0), o da qualche parte in mezzo
-   I gate sono **dinamici**: il loro valore è calcolato basato sul
    contesto corrente

### RNN Bidirezionali e Multi-layer

**Motivazione:** Classificazione del Sentiment

-   Possiamo considerare l'hidden state come una rappresentazione della
    parola nel contesto della frase --- chiamiamo questa una
    **rappresentazione contestuale**
-   Queste rappresentazioni contestuali contengono solo informazione sul
    **contesto sinistro** (es. 'the movie was')
-   Cosa riguardo al contesto destro?
-   In questo esempio, 'exciting' è nel contesto destro e modifica il
    significato di 'terribly' (da negativo a positivo)

------------------------------------------------------------------------

## 🌐 Neural Machine Translation (NMT)

La **Neural Machine Translation (NMT)** è un modo per fare Machine
Translation con una **singola rete neurale end-to-end**.

-   L'architettura della rete neurale è chiamata **sequence-to-sequence
    model** (aka seq2seq) e coinvolge **due RNN**

### Il Modello Sequence-to-Sequence

-   **Encoder RNN** produce un'encoding della frase sorgente
-   **Decoder RNN** genera la frase target

### Il Problema del Bottleneck

**Problema con questa architettura?** L'encoding della frase sorgente è
un **bottleneck** --- tutta l'informazione della frase sorgente deve
essere compressa in un singolo vettore.

------------------------------------------------------------------------

## 👀 Attention

**L'Attention fornisce una soluzione al problema del bottleneck.**

**Core idea:** a ogni passo del decoder, usa una connessione diretta
all'encoder per **focalizzarsi su una parte particolare** della sequenza
sorgente.

### Attention: in Equazioni

Abbiamo encoder hidden states $h_1, \dots, h_N \in \mathbb{R}^h$. Al
timestep $t$, abbiamo decoder hidden state $s_t$.

Otteniamo gli **attention scores** per questo passo:

$$e^t = [s_t^T h_1, \dots, s_t^T h_N] \in \mathbb{R}^N$$

Prendiamo la softmax per ottenere la **distribuzione di attention** per
questo passo (è una distribuzione di probabilità e somma a 1):

$$\alpha^t = \text{softmax}(e^t) \in \mathbb{R}^N$$

Usiamo $\alpha^t$ per prendere una **somma pesata** degli encoder hidden
states per ottenere l'**attention output**:

$$a_t = \sum_{i=1}^{N} \alpha_i^t h_i \in \mathbb{R}^h$$

Infine concateniamo l'attention output con il decoder hidden state e
procediamo come nel modello seq2seq senza attention:

$$[a_t; s_t] \in \mathbb{R}^{2h}$$

------------------------------------------------------------------------

## ⚠️ Problemi con i Modelli Ricorrenti

### Distanza di Interazione Lineare

-   Le RNN sono unrolled "da sinistra a destra"
-   Questo codifica **località lineare**: un'euristica utile! Le parole
    vicine spesso influenzano i significati l'una dell'altra
-   **Problema:** le RNN impiegano $O(\text{sequence length})$ passi
    perché coppie di parole distanti interagiscano

------------------------------------------------------------------------

## 🔍 Self-Attention

**Ricorda:** l'Attention opera su queries, keys e values.

-   Abbiamo alcune queries $q_1, q_2, \dots, q_T$. Ogni query è
    $q_i \in \mathbb{R}^d$
-   Abbiamo alcune keys $k_1, k_2, \dots, k_T$. Ogni key è
    $k_i \in \mathbb{R}^d$
-   Abbiamo alcuni values $v_1, v_2, \dots, v_T$. Ogni value è
    $v_i \in \mathbb{R}^d$

Nella **self-attention**, le queries, keys e values sono **disegnate
dalla stessa sorgente**. Per esempio, se l'output del layer precedente è
$x_1, \dots, x_T$, potremmo lasciare $v_i = k_i = q_i = x_i$ (usare gli
stessi vettori per tutti!)

L'operazione di self-attention (dot product) è la seguente:

$$e_{ij} = q_i^T k_j \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})} \qquad \text{output}_i = \sum_j \alpha_{ij} v_j$$

1.  Calcola le affinità key-query
2.  Calcola i pesi di attention dalle affinità (softmax)
3.  Calcola gli output come somma pesata dei values

------------------------------------------------------------------------

## 🏗️ Il Transformer

### Transformer Encoder: Key-Query-Value Attention

Il Transformer fa la self-attention in un modo particolare:

-   Siano $x_1, \dots, x_T$ i vettori di input al Transformer encoder;
    $x_i \in \mathbb{R}^d$
-   Allora keys, queries, values sono:
    -   $k_i = Kx_i$, dove $K \in \mathbb{R}^{d \times d}$ è la **key
        matrix**
    -   $q_i = Qx_i$, dove $Q \in \mathbb{R}^{d \times d}$ è la **query
        matrix**
    -   $v_i = Vx_i$, dove $V \in \mathbb{R}^{d \times d}$ è la **value
        matrix**
-   Queste matrici permettono a diversi aspetti dei vettori $x$ di
    essere usati/enfatizzati in ciascuno dei tre ruoli

### Multi-Headed Attention

Cosa se vogliamo guardare in **più posti** nella frase
contemporaneamente?

-   Per la parola $i$, la self-attention "guarda" dove $x_i^T Q^T Kx_j$
    è alto, ma forse vogliamo focalizzarci su diversi $j$ per diverse
    ragioni?
-   Definiamo **multiple attention "heads"** attraverso multiple Q, K, V
    matrices
-   Siano $Q_\ell, K_\ell, V_\ell \in \mathbb{R}^{d \times d/h}$, dove
    $h$ è il numero di attention heads, e $\ell$ varia da 1 a $h$
-   Ogni attention head esegue l'attention **indipendentemente**:

$$\text{output}_\ell = \text{softmax}(XQ_\ell K_\ell^T X^T) * XV_\ell$$

-   Poi gli output di tutte le heads sono **combinati**:

$$\text{output} = Y[\text{output}_1; \dots; \text{output}_h]$$

-   Ogni head può "guardare" cose diverse e costruire value vectors
    diversamente

### Scaled Dot Product

Quando la dimensionalità $d$ diventa grande, i dot products tra vettori
tendono a diventare grandi. A causa di questo, gli input alla softmax
possono essere grandi, rendendo i gradienti piccoli.

Dividiamo gli attention scores per $\sqrt{d/h}$ per fermare i punteggi
dal diventare grandi solo come funzione di $d/h$:

$$\text{output}_\ell = \text{softmax}\left(\frac{XQ_\ell K_\ell^T X^T}{\sqrt{d/h}}\right) * XV_\ell$$

### Residual Connections

Le **residual connections** sono un trucco per aiutare i modelli ad
addestrarsi meglio.

-   Invece di $X^{(i)} = \text{Layer}(X^{(i-1)})$
-   Lasciamo $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$ (così
    dobbiamo solo apprendere "il residuo" dal layer precedente)
-   Le residual connections sono pensate per rendere il loss landscape
    considerevolmente più liscio (quindi training più facile!)

### Layer Normalization

La **layer normalization** è un trucco per aiutare i modelli ad
addestrarsi più velocemente.

-   **Idea:** ridurre la variazione non informativa nei valori dei
    vettori hidden normalizzando a media e deviazione standard unitarie
    entro ogni layer
-   Sia $x \in \mathbb{R}^d$ un vettore (parola) individuale nel modello
-   Sia $\mu = \frac{1}{d} \sum_{j=1}^{d} x_j$ la media
-   Sia $\sigma = \sqrt{\frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2}$ la
    deviazione standard
-   Siano $\gamma \in \mathbb{R}^d$ e $\beta \in \mathbb{R}^d$ parametri
    appresi di "gain" e "bias"
-   La layer normalization calcola:
    $\frac{x - \mu}{\sigma} * \gamma + \beta$

### Codifica della Posizione

Poiché la self-attention **non costruisce informazione sull'ordine**,
dobbiamo codificare l'ordine della frase nelle nostre keys, queries e
values.

-   Considera rappresentare ogni indice di sequenza come un vettore
    $p_i \in \mathbb{R}^d$ --- questi sono **position vectors**
-   Facile da incorporare: basta **aggiungere** i $p_i$ ai nostri input!
-   Siano $\tilde{v}_i, \tilde{k}_i, \tilde{q}_i$ i nostri vecchi
    values, keys e queries:

$$v_i = \tilde{v}_i + p_i, \qquad q_i = \tilde{q}_i + p_i, \qquad k_i = \tilde{k}_i + p_i$$

In reti self-attention profonde, facciamo questo al **primo layer**!

``` mermaid
flowchart LR
    A[Input sequence] --> B[Positional Encoding]
    B --> C[Multi-Head Self-Attention]
    C --> D[Residual + LayerNorm]
    D --> E[Feed-Forward]
    E --> F[Residual + LayerNorm]
    F --> G[Output]
```

### Benchmark GLUE

Il **GLUE** (General Language Understanding Evaluation) è un benchmark
standard per valutare i modelli di linguaggio: - Raccolta di task di
comprensione del linguaggio naturale (classificazione di frasi,
entailment, question answering, ecc.) - Permette di confrontare modelli
Transformer e altri modelli NLP su una varietà di compiti - I modelli
pre-addestrati (come BERT) sono tipicamente valutati su GLUE

------------------------------------------------------------------------

## 🌊 Normalizing Flows

I **Normalizing Flows** sono una famiglia di modelli generativi che
trasformano una distribuzione semplice (es. Gaussiana) in una
distribuzione complessa attraverso una **sequenza di trasformazioni
invertibili**:

$$z \xrightarrow{f_1} h_1 \xrightarrow{f_2} h_2 \xrightarrow{f_3} \dots \xrightarrow{f_K} x$$

-   Ogni trasformazione $f_k$ deve essere **invertibile** e
    **differenziabile**
-   La densità si calcola con il **change of variables**:

$$p(x) = p(z) \prod_{k=1}^{K} \left|\det \frac{\partial f_k^{-1}}{\partial h_{k-1}}\right|$$

-   Permettono **inferenza esatta** (a differenza di VAE e GAN)
-   Usati per density estimation e generazione di dati

------------------------------------------------------------------------

## 🌳 Neural Dependency Parsing

Il **dependency parsing** è il task di analizzare la **struttura
sintattica** di una frase, rappresentando le relazioni di dipendenza tra
le parole (es. soggetto, oggetto, modificatore).

### Il Modello di Chen e Manning (2014)

Il modello di **Chen e Manning (2014)** è un parser sintattico basato su
**architetture neurali**: - Usa un **transition-based parser** che
costruisce l'albero di dipendenza tramite una sequenza di transizioni -
Le transizioni possibili sono: **Shift**, **Left-Arc$_r$**,
**Right-Arc$_r$** - Una rete neurale predice la prossima transizione
data la configurazione corrente

### Metriche di Valutazione

  -----------------------------------------------------------------------
  Metrica                       Descrizione
  ----------------------------- -----------------------------------------
  **UAS** (Unlabeled Attachment Percentuale di parole con l'**arco**
  Score)                        corretto (senza considerare l'etichetta)

  **LAS** (Labeled Attachment   Percentuale di parole con arco **e**
  Score)                        etichetta corretti
  -----------------------------------------------------------------------

Questi vengono confrontati con parser tradizionali come **MaltParser**,
**MSTParser** e **TurboParser**.

### Rappresentazioni Distribuite Concatenate

Il modello usa **vettori densi** ($d$-dimensionali) non solo per le
parole (*word embeddings*), ma anche per: - I **tag POS**
(Part-of-Speech) --- es. prossimità vettoriale tra `NNS` e `NN` - Le
**etichette di dipendenza** --- es. prossimità tra `nummod` e `amod`

### Rappresentazione della Configurazione

La configurazione del parser è rappresentata estraendo e concatenando le
componenti vettoriali dallo **Stack** e dal **Buffer**:

$$s_1, s_2, b_1, lc(s_1), rc(s_1), \dots$$

per **parola**, **POS** e **relazione di dipendenza**.

### Architettura della Rete

1.  **Strato di Input:** lookup e concatenazione dei vettori
2.  **Strato nascosto non lineare** con attivazione **ReLU**:

$$h = \text{ReLU}(Wx + b_1)$$

3.  **Strato di output Softmax** per predire le transizioni tra
    $\{Shift, Left\text{-}Arc_r, Right\text{-}Arc_r\}$:

$$y = \text{softmax}(Uh + b_2)$$

``` mermaid
flowchart LR
    A[Configurazione<br/>Stack + Buffer] --> B[Input layer<br/>lookup + concat]
    B --> C[Hidden layer<br/>ReLU]
    C --> D[Output layer<br/>Softmax]
    D --> E[Transizione<br/>Shift / Left-Arc / Right-Arc]
```

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 15_reinforcement_learning.md ===== -->
```

------------------------------------------------------------------------

# PARTE VII --- Reinforcement Learning

# 15 --- Reinforcement Learning (RL)

> *Riferimento: Roger Grosse CSC2515 · Emma Brunskill CS234 Stanford ·
> Sutton & Barto*

------------------------------------------------------------------------

## 🎮 Cos'è il Reinforcement Learning

Il Reinforcement Learning è l'apprendimento **attraverso
esperienza/dati** per prendere buone decisioni sotto incertezza.

Ricordiamo che abbiamo categorizzato i tipi di ML in base a quanta
informazione forniscono sul comportamento desiderato: - **Supervised
learning:** etichette del comportamento desiderato - **Unsupervised
learning:** nessuna etichetta - **Reinforcement learning:** segnale di
ricompensa che valuta l'esito delle azioni passate

In RL, ci concentriamo tipicamente sul **processo decisionale
sequenziale**: un agente sceglie una sequenza di azioni che influenzano
ciascuna le possibilità future disponibili all'agente.

``` mermaid
flowchart LR
    A[Agente] -->|osserva il mondo| B[Stato]
    B -->|prende un'azione| C[Cambio di stato]
    C -->|ricompensa| A
```

### Il RL Generalmente Coinvolge

-   **Ottimizzazione**
-   **Conseguenze ritardate**
-   **Esplorazione**
-   **Generalizzazione**

------------------------------------------------------------------------

## 🧮 Markov Decision Process (MDP)

La maggior parte del RL è fatta in un framework matematico chiamato
**Markov Decision Process (MDP)**.

### Stati e Azioni

-   Lo **stato** è una descrizione dell'ambiente in dettaglio
    sufficiente per determinare la sua evoluzione
-   **Markov assumption:** lo stato al tempo $t+1$ dipende direttamente
    dallo stato e dall'azione al tempo $t$, ma non da stati e azioni
    passati
-   Per descrivere la dinamica, dobbiamo specificare le **probabilità di
    transizione** $P(S_{t+1} \mid S_t, A_t)$
-   In questa lezione, assumiamo che lo stato sia **completamente
    osservabile**, un'assunzione altamente non banale

### Politiche (Policies)

Il modo in cui l'agente sceglie l'azione in ogni passo è chiamato
**policy**. Considereremo due tipi: - **Deterministic policy:**
$A_t = \pi(S_t)$ per qualche funzione $\pi: S \to A$ - **Stochastic
policy:** $A_t \sim \pi(\cdot \mid S_t)$ per qualche funzione
$\pi: S \to P(A)$

Con le politiche stocastiche, la distribuzione sui rollout (o
traiettorie) si fattorizza:

$$p(s_1, a_1, \dots, s_T, a_T) = p(s_1) \pi(a_1 \mid s_1) P(s_2 \mid s_1, a_1) \pi(a_2 \mid s_2) \cdots P(s_T \mid s_{T-1}, a_{T-1}) \pi(a_T \mid s_T)$$

> Il fatto che le politiche debbano considerare solo lo stato corrente è
> una potente conseguenza della **Markov assumption** e della piena
> osservabilità. Se l'ambiente è parzialmente osservabile, la policy
> deve dipendere dalla storia delle osservazioni.

### Ricompense (Rewards)

A ogni passo temporale, l'agente riceve una ricompensa da una
distribuzione che dipende dallo stato e dall'azione correnti:

$$R_t \sim \mathcal{R}(\cdot \mid S_t, A_t)$$

Per semplicità, assumiamo che le ricompense siano deterministiche, cioè
$R_t = r(S_t, A_t)$.

Il **return** determina quanto è stato buono l'esito di un episodio: -
**Undiscounted:** $G = R_0 + R_1 + R_2 + \cdots$ - **Discounted:**
$G = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots$

L'obiettivo è **massimizzare il return atteso**, $E[G]$.

-   $\gamma$ è un iperparametro chiamato **discount factor** che
    determina quanto ci importa delle ricompense ora vs. ricompense dopo

### Definizione Formale di MDP

Un **Markov Decision Process (MDP)** è definito da una tupla
$(S, A, P, R, \gamma)$:

-   $S$: **State space** (discreto o continuo)
-   $A$: **Action space** (qui consideriamo spazio di azioni finito,
    cioè $A = \{a_1, \dots, a_{|A|}\}$)
-   $P$: **Transition probability**
-   $R$: **Immediate reward distribution**
-   $\gamma$: **Discount factor** ($0 \le \gamma < 1$)

Insieme questi definiscono l'ambiente in cui l'agente opera, e gli
obiettivi che dovrebbe raggiungere.

------------------------------------------------------------------------

## 📈 Value Functions

### State Value Function

La **value function** $V^\pi$ per una policy $\pi$ misura il return
atteso se inizi nello stato $s$ e segui la policy $\pi$:

$$V^\pi(s) \triangleq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k} \mid S_t = s\right]$$

Questo misura la **desiderabilità dello stato** $s$.

### Equazioni di Bellman

La fondazione di molti algoritmi RL è il fatto che le value functions
soddisfano una **relazione ricorsiva**, chiamata **equazione di
Bellman**:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_t + \gamma G_{t+1} \mid S_t = s]$$

$$= \sum_a \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s'} P(s' \mid a, s) V^\pi(s')\right]$$

Vedendo $V^\pi$ come un vettore (dove le voci corrispondono agli stati),
definiamo l'**operatore di Bellman backup** $T^\pi$:

$$(T^\pi V)(s) \triangleq \sum_a \pi(a \mid s) \left[r(s, a) + \gamma \sum_{s'} P(s' \mid a, s) V(s')\right]$$

L'equazione di Bellman può essere vista come un **punto fisso**
dell'operatore di Bellman:

$$T^\pi V^\pi = V^\pi$$

### State-Action Value Function (Q-Function)

Una funzione strettamente correlata ma utilmente diversa è la
**state-action value function**, o **Q-function**, $Q^\pi$ per la policy
$\pi$, definita come:

$$Q^\pi(s, a) \triangleq \mathbb{E}_\pi\left[\sum_{k \geq 0} \gamma^k R_{t+k} \mid S_t = s, A_t = a\right]$$

**Se conoscessi $Q^\pi$, come otterresti $V^\pi$?**

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

**Se conoscessi $V^\pi$, come otterresti $Q^\pi$?** Applica un'equazione
simile a Bellman:

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s'} P(s' \mid a, s) V^\pi(s')$$

Questo richiede di conoscere la dinamica, quindi in generale non è
facile recuperare $Q^\pi$ da $V^\pi$.

### Optimal State-Action Value Function

Supponi di essere nello stato $s$. Puoi scegliere un'azione $a$, e poi
seguire la (fissa) policy $\pi$ da lì in poi. Cosa scegli?

$$\arg\max_a Q^\pi(s, a)$$

Se una policy deterministica $\pi$ è ottimale, allora deve essere il
caso che per qualsiasi stato $s$:

$$\pi(s) = \arg\max_a Q^\pi(s, a)$$

altrimenti potresti migliorare la policy cambiando $\pi(s)$.

------------------------------------------------------------------------

## 🔍 Trovare la Value Function Ottimale: Value Iteration

Usiamo la programmazione dinamica per trovare $Q^*$.

**Value Iteration:** parti da una funzione iniziale $Q_1$. Per ogni
$k = 1, 2, \dots$, applica:

$$Q_{k+1} \leftarrow T^* Q_k$$

Scrivendo l'aggiornamento per esteso:

$$Q_{k+1}(s, a) \leftarrow r(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \max_{a' \in A} Q_k(s', a')$$

> Osserva: un punto fisso di questo aggiornamento è esattamente una
> soluzione dell'equazione di Bellman ottimale, che caratterizza la
> Q-function di una policy ottimale.

### Policy Iteration

Un altro metodo di programmazione dinamica è la **policy iteration**,
che alterna due passi:

1.  **Policy Evaluation:** calcola la value function $V^\pi$ della
    policy corrente $\pi$ (iterando l'equazione di Bellman)
2.  **Policy Improvement:** migliora la policy scegliendo greedy
    rispetto a $V^\pi$:

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

``` mermaid
flowchart LR
    A[Policy Evaluation<br/>calcola V^π] --> B[Policy Improvement<br/>π' greedy su V^π]
    B --> C{Policy cambiata?}
    C -->|Sì| A
    C -->|No| D[Policy ottimale]
```

-   La policy iteration **converge** alla policy ottimale
-   Il miglioramento è **monotono**: ogni iterazione produce una policy
    migliore o uguale

------------------------------------------------------------------------

## 🎲 Verso l'Apprendimento

Ora concentriamoci sul reinforcement learning, dove l'ambiente è
**sconosciuto**. Come possiamo applicare l'apprendimento?

1.  **Apprendi un modello dell'ambiente**, e fai planning nel modello
    (cioè model-based RL)
    -   Sai già come fare questo in principio, ma è molto difficile
        farlo funzionare
2.  **Apprendi una value function** (es. Q-learning)
3.  **Apprendi una policy direttamente** (es. policy gradient)

**Come gestire spazi di stati estremamente grandi?** - **Function
approximation:** scegli una forma parametrica per la policy e/o la value
function (es. lineare nelle feature, rete neurale, ecc.)

### Monte Carlo Estimation

Ricorda l'equazione di Bellman ottimale:

$$Q^*(s, a) = r(s, a) + \gamma \mathbb{E}_{P(s' \mid s, a)}\left[\max_{a'} Q^*(s', a')\right]$$

**Problema:** dobbiamo conoscere la dinamica per valutare l'aspettativa.

**Monte Carlo estimation** di un'aspettativa $\mu = E[X]$: campiona
ripetutamente $X$ e aggiorna:

$$\mu \leftarrow \mu + \alpha(X - \mu)$$

**Idea:** applica la Monte Carlo estimation all'equazione di Bellman
campionando $S' \sim P(\cdot \mid s, a)$ e aggiornando:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[\underbrace{r(s, a) + \gamma \max_{a'} Q(S', a')}_{} - Q(s, a)\right]$$

> Questo è un esempio di **temporal difference learning**, cioè
> aggiornare le nostre predizioni per farle corrispondere alle nostre
> predizioni successive (una volta che abbiamo più informazione).

### Temporal Difference (TD) Learning

Il **Temporal Difference (TD) learning** è un metodo per stimare la
value function senza conoscere il modello dell'ambiente.

**TD(0) per stimare $V^\pi$:**

$$V(s_t) \leftarrow V(s_t) + \alpha\left[r_t + \gamma V(s_{t+1}) - V(s_t)\right]$$

-   Il termine $r_t + \gamma V(s_{t+1})$ è chiamato **TD target**
-   Il termine $r_t + \gamma V(s_{t+1}) - V(s_t)$ è chiamato **TD
    error**
-   A differenza del Monte Carlo, il TD learning **aggiorna dopo ogni
    passo** (bootstrap), non alla fine dell'episodio
-   **Vantaggio:** può apprendere da episodi incompleti, più efficiente
    in campioni

### Monte Carlo vs TD Learning

                            Monte Carlo               TD Learning
  ------------------------- ------------------------- -----------------
  **Aggiornamento**         Alla fine dell'episodio   Dopo ogni passo
  **Bootstrap**             No                        Sì
  **Varianza**              Alta                      Bassa
  **Bias**                  Basso                     Alto
  **Efficienza campioni**   Bassa                     Alta

------------------------------------------------------------------------

## 🎯 Q-Learning con ε-Greedy Policy

**Parametri:** - Learning rate $\alpha$ - Exploration parameter
$\varepsilon$ - Inizializza $Q(s, a)$ per tutti $(s, a) \in S \times A$

L'agente parte allo stato $S_0$. Per il passo temporale
$t = 0, 1, \dots$:

1.  Scegli $A_t$ secondo la **ε-greedy policy**:

$$A_t \leftarrow \begin{cases} \arg\max_{a \in A} Q(S_t, a) & \text{con probabilità } 1 - \varepsilon \\ \text{azione uniformemente casuale in } A & \text{con probabilità } \varepsilon \end{cases}$$

2.  Prendi l'azione $A_t$ nell'ambiente
3.  Lo stato cambia da $S_t$ a $S_{t+1} \sim P(\cdot \mid S_t, A_t)$
4.  Osserva $S_{t+1}$ e $R_t$
5.  Aggiorna la action-value function allo state-action $(S_t, A_t)$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_t + \gamma \max_{a' \in A} Q(S_{t+1}, a') - Q(S_t, A_t)\right]$$

------------------------------------------------------------------------

## ⚖️ Esplorazione vs Sfruttamento

La **ε-greedy** è un semplice meccanismo per mantenere il tradeoff
esplorazione-sfruttamento:

$$\pi_\varepsilon(S; Q) = \begin{cases} \arg\max_{a \in A} Q(S, a) & \text{con probabilità } 1 - \varepsilon \\ \text{azione uniformemente casuale in } A & \text{con probabilità } \varepsilon \end{cases}$$

-   La ε-greedy policy garantisce che la maggior parte del tempo
    (probabilità $1 - \varepsilon$) l'agente **sfrutti** la sua
    conoscenza incompleta del mondo scegliendo la migliore azione (cioè
    quella corrispondente alla più alta action-value), ma
    occasionalmente (probabilità $\varepsilon$) **esplora** altre azioni
-   **Senza esplorazione**, l'agente potrebbe non trovare mai alcune
    buone azioni
-   La ε-greedy è uno dei metodi più semplici, ma ampiamente usati, per
    il tradeoff esplorazione-sfruttamento

------------------------------------------------------------------------

## 🔓 Off-Policy Learning

**Q-learning update di nuovo:**

$$Q(S, A) \leftarrow Q(S, A) + \alpha\left[R + \gamma \max_{a' \in A} Q(S', a') - Q(S, A)\right]$$

**Nota:** questo aggiornamento non menziona la policy da nessuna parte.
L'unica cosa per cui la policy è usata è determinare quali stati sono
visitati.

-   Questo significa che possiamo seguire **qualsiasi policy** vogliamo
    (es. ε-greedy), e converge comunque alla Q-function ottimale
-   Algoritmi come questo sono noti come **off-policy algorithms**, e
    questa è una proprietà estremamente utile
-   Il **policy gradient** è un algoritmo **on-policy**. Incoraggiare
    l'esplorazione è molto più difficile in quel caso

------------------------------------------------------------------------

## 🧠 Function Approximation

Finora, abbiamo assunto una **rappresentazione tabulare** di $Q$: una
voce per ogni coppia stato/azione.

-   Questo è impraticabile da memorizzare per tutti tranne i problemi
    più semplici, e non condivide struttura tra stati correlati
-   **Soluzione:** approssima $Q$ usando una funzione parametrizzata,
    es.:
    -   **Linear function approximation:** $Q(s, a) = w^T \psi(s, a)$
    -   Calcola $Q$ con una **rete neurale**
-   Aggiorna $Q$ usando backprop:

$$t \leftarrow r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a)$$

$$\theta \leftarrow \theta + \alpha(t - Q(s, a)) \nabla_\theta Q(s_t, a_t)$$

### Deep Q-Learning (Atari)

Approssimare $Q$ con una rete neurale è un'idea vecchia di decenni, ma
DeepMind l'ha fatta funzionare molto bene sui giochi Atari nel 2013
("deep Q-learning").

-   Usavano una rete molto piccola per gli standard di oggi
-   **Principale innovazione tecnica:** memorizzare l'esperienza in un
    **replay buffer**, ed eseguire Q-learning usando l'esperienza
    memorizzata
-   Guadagna efficienza campionaria **separando l'interazione con
    l'ambiente dall'ottimizzazione** --- non serve nuova esperienza per
    ogni aggiornamento SGD!

**Algoritmo:** 1. Prendi qualche azione $a$; osserva
$(s_i, a_i, s', r_i)$, aggiungila a $B$ 2. Campiona mini-batch
$\{s, a, s', r\}$ da $B$ uniformemente 3. Calcola
$y_j = r + \gamma \max_{a'} Q(s', a')$ usando la **target network** $Q'$
4. Aggiorna $\phi \leftarrow \phi - \alpha \sum_j \frac{dQ_\phi}{d\phi}$
5. Aggiorna $\phi'$: copia $\phi$ ogni $N$ passi

**Risultati (Mnih et al., Nature 2015):** - La rete riceveva **pixel
grezzi** come osservazioni - La stessa architettura condivisa tra tutti
i giochi - Dopo circa un giorno di training su un particolare gioco,
spesso batteva la performance "a livello umano" - Andava molto bene sui
giochi reattivi, male su quelli che richiedono pianificazione (es.
Montezuma's Revenge)

------------------------------------------------------------------------

## 📊 Recap e Altri Approcci

-   Tutti gli approcci discussi stimano prima la value function. Sono
    chiamati **value-based methods**
-   Ci sono metodi che **ottimizzano direttamente la policy**, cioè
    **policy search methods**
-   I **model-based RL** methods stimano il vero (ma sconosciuto)
    modello dell'ambiente $P$ con una stima $\hat{P}$, e usano la stima
    $\hat{P}$ per pianificare
-   Ci sono **metodi ibridi**

``` mermaid
flowchart TB
    A[RL Agents] --> B[Model-based<br/>esplicito: modello]
    A --> C[Model-free<br/>esplicito: value/policy]
    C --> D[Value-based<br/>Q-learning]
    C --> E[Policy-based<br/>policy gradient]
    B --> F[Planning nel modello]
```

------------------------------------------------------------------------

## 🧩 Tipi di Agenti RL

  ---------------------------------------------------------------------------
  Tipo              Esplicito                  Descrizione
  ----------------- -------------------------- ------------------------------
  **Model-based**   Modello                    Può avere o meno policy e/o
                                               value function

  **Model-free**    Value function e/o policy  Nessun modello
                    function                   
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

## 📐 Markov Process e Markov Reward Process

### Markov Process (Markov Chain)

Processo casuale **senza memoria** --- sequenza di stati casuali con
proprietà di Markov.

**Definizione di Markov Process:** - $S$ è un insieme (finito) di stati
($s \in S$) - $P$ è la dinamica/modello di transizione che specifica
$p(s_{t+1} = s' \mid s_t = s)$ - **Nota:** niente ricompense, niente
azioni - Se c'è un numero finito ($N$) di stati, possiamo esprimere $P$
come una matrice

### Markov Reward Process (MRP)

Un **Markov Reward Process** è una Markov Chain + ricompense.

**Definizione di MRP:** - $S$ è un insieme (finito) di stati
($s \in S$) - $R$ è una reward function
$R(s_t = s) = E[r_t \mid s_t = s]$ - $P$ è la dinamica/modello di
transizione che specifica $P(s_{t+1} = s' \mid s_t = s)$ - Discount
factor $\gamma \in [0, 1]$ - **Nota:** niente azioni

### Return e Value Function

**Definizione di Horizon ($H$):** numero di passi temporali in ogni
episodio. Può essere infinito. Altrimenti chiamato **finite Markov
reward process**.

**Definizione di Return, $G_t$** (per un MRP): somma scontata delle
ricompense dal passo temporale $t$ all'orizzonte $H$:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{H-1} r_{t+H-1}$$

**Definizione di State Value Function, $V(s)$** (per un MRP): return
atteso partendo dallo stato $s$:

$$V(s) = E[G_t \mid s_t = s] = E[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{H-1} r_{t+H-1} \mid s_t = s]$$

### Discount Factor

-   **Matematicamente conveniente** (evita returns e values infiniti)
-   Gli umani spesso agiscono come se ci fosse un discount factor \< 1
-   Se le lunghezze degli episodi sono sempre finite ($H < \infty$),
    possiamo usare $\gamma = 1$

### Calcolare il Valore di un MRP

La proprietà di Markov fornisce struttura. La MRP value function
soddisfa:

$$V(s) = \underbrace{R(s)}_{} + \underbrace{\gamma \sum_{s' \in S} P(s' \mid s) V(s')}_{\text{Discounted sum of future rewards}}$$

------------------------------------------------------------------------

## 🧭 Evaluation e Control

-   **Evaluation:** stima/predice le ricompense attese dal seguire una
    data policy
-   **Control:** ottimizzazione --- trova la migliore policy

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 16_apprendimento_relazionale.md ===== -->
```

------------------------------------------------------------------------

# PARTE VIII --- Apprendimento relazionale e rappresentazioni

# 16 --- Apprendimento Relazionale e Inductive Logic Programming (ILP)

> *Riferimento: De Raedt, Logical and Relational Learning: From ILP to
> MRDM*

------------------------------------------------------------------------

## 🧩 Introduzione

L'**apprendimento relazionale** (Relational Learning) si occupa di
apprendere modelli che coinvolgono **entità e relazioni articolate**,
rappresentate usando la **logica del primo ordine (FOL)**.

-   A differenza dell'apprendimento proposizionale (attributo-valore),
    qui gli esempi possono avere **struttura** e **relazioni** tra loro
-   Le tecniche per apprendere programmi logici provengono dall'area
    dell'**Inductive Logic Programming (ILP)**

------------------------------------------------------------------------

## 🔁 Inductive Logic Programming (ILP)

L'**ILP** è l'area che studia come apprendere **programmi logici**
(insiemi di regole) dai dati.

-   Una definizione ricorsiva può essere vista come un **programma
    logico**
-   Le tecniche per apprendere programmi logici provengono dall'area
    dell'ILP

### Il Problema della Ricorsione

-   Le definizioni **ricorsive** sono difficili da apprendere
-   Inoltre, pochi problemi pratici richiedono la ricorsione
-   Quindi: molte tecniche ILP sono **ristrette a definizioni non
    ricorsive** per rendere l'apprendimento più facile

``` mermaid
flowchart LR
    A[Programma logico] --> B{Ricorsivo?}
    B -->|Sì| C[Difficile da apprendere]
    B -->|No| D[Più facile da apprendere]
    C --> E[Pochi problemi pratici richiedono ricorsione]
```

------------------------------------------------------------------------

## 🧠 Rappresentazione della Conoscenza

La **logica del primo ordine (FOL)** permette di modellare: - **Entità**
(oggetti, individui) - **Relazioni** tra entità (predicati) -
**Proprietà** delle entità

**Esempio:** una relazione "padre di" può essere rappresentata come un
predicato `padre(X, Y)` che lega due individui.

------------------------------------------------------------------------

## 🔍 Spazio di Ricerca e Generalizzazione

### Algoritmi Generate-and-Test

Gli algoritmi di apprendimento relazionale usano tipicamente un
approccio **generate-and-test**: 1. **Genera** una regola candidata
(ipotesi) 2. **Testa** la regola sui dati (esempi positivi e negativi)
3. Se la regola copre bene gli esempi, la mantieni; altrimenti la
raffinì

### Struttura dello Spazio di Ricerca

-   Lo spazio di ricerca è l'insieme di tutte le possibili
    regole/ipotesi
-   Le regole sono ordinate per **generalità** (una regola più generale
    copre più esempi)

### Proprietà di Monotonicità

-   L'operatore di **raffinamento** (refinement) trasforma una regola in
    una più specifica (aggiungendo condizioni) o più generale
    (rimuovendo condizioni)
-   La monotonicità garantisce che raffinare una regola **non aumenti**
    il numero di esempi coperti

### Operatori di Raffinamento

-   **Specializzazione:** aggiungere un letterale alla premessa della
    regola (riduce la copertura)
-   **Generalizzazione:** rimuovere un letterale dalla premessa (aumenta
    la copertura)

------------------------------------------------------------------------

## 🧩 Concept Learning con Description Logics (DL)

Le **Description Logics (DL)** sono una famiglia di linguaggi di
rappresentazione della conoscenza basati sulla logica, usati per
modellare **ontologie** (es. OWL).

### Definizione del Problema

Il **problema di apprendimento di concetti** in DL consiste nel trovare
una descrizione di concetto $C$ tale che: - Copra tutti gli **esempi
positivi** (individui che appartengono al concetto) - Escluda tutti gli
**esempi negativi** (individui che non appartengono al concetto)

### Algoritmo DLFoil

**DLFoil** è un algoritmo per apprendere concetti in Description
Logics: - Usa un approccio **separate-and-conquer** (come PRISM) -
Genera una descrizione di concetto aggiungendo **restrizioni** (role
restrictions) che massimizzano la copertura degli esempi positivi -
Raffina iterativamente la descrizione finché non copre tutti gli esempi
positivi

``` mermaid
flowchart TB
    A[Esempi positivi e negativi] --> B[Genera descrizione di concetto]
    B --> C[Aggiungi restrizioni che massimizzano copertura]
    C --> D{Tutti i positivi coperti?}
    D -->|No| C
    D -->|Sì| E[Concetto appreso]
```

------------------------------------------------------------------------

## 📊 Riepilogo

  Concetto                 Descrizione
  ------------------------ ----------------------------------------------------------
  **ILP**                  Apprendere programmi logici dai dati
  **FOL**                  Logica del primo ordine per modellare entità e relazioni
  **Generate-and-test**    Genera ipotesi, testa sui dati
  **Raffinamento**         Operatori per specializzare/generalizzare regole
  **Description Logics**   Linguaggi per ontologie (es. OWL)
  **DLFoil**               Algoritmo per apprendere concetti in DL

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 17_knowledge_graph_embeddings.md ===== -->
```
# 17 --- Knowledge Graph Embeddings (KGE)

> *Riferimento: Knowledge Graph Embeddings --- rappresentazione di grafi
> di conoscenza in spazi vettoriali*

------------------------------------------------------------------------

## 🧩 Introduzione

I **Knowledge Graph Embeddings (KGE)** sono tecniche per rappresentare
entità e relazioni di un **grafo di conoscenza** in **spazi vettoriali a
bassa dimensione**.

-   Permettono di applicare tecniche di machine learning a dati
    strutturati (grafi)
-   Catturano la **semantica** delle entità e delle relazioni

------------------------------------------------------------------------

## 📊 Rappresentazione dei Grafi di Conoscenza

Un **grafo di conoscenza** è modellato come un insieme di **triple**
$(s, p, o)$:

-   **s** = soggetto (subject)
-   **p** = predicato/relazione (predicate)
-   **o** = oggetto (object)

**Esempio:** `(Roma, capitale_di, Italia)` --- Roma è soggetto,
`capitale_di` è la relazione, Italia è l'oggetto.

### Standard Semantici

  -----------------------------------------------------------------------
  Standard                       Descrizione
  ------------------------------ ----------------------------------------
  **RDF** (Resource Description  Modello base per rappresentare triple
  Framework)                     

  **RDFS** (RDF Schema)          Estende RDF con classi e proprietà

  **OWL** (Web Ontology          Linguaggio per ontologie più espressive
  Language)                      
  -----------------------------------------------------------------------

``` mermaid
flowchart LR
    A[Roma] -->|capitale_di| B[Italia]
    C[Parigi] -->|capitale_di| D[Francia]
```

------------------------------------------------------------------------

## 🎯 Modelli KGE

I modelli KGE mappano **entità** e **relazioni** in spazi vettoriali a
bassa dimensione, in modo che le triple valide siano rappresentate da
relazioni vettoriali coerenti.

### RESCAL

**RESCAL** rappresenta le relazioni come **matrici** che trasformano il
vettore del soggetto nel vettore dell'oggetto:

$$f(s, p, o) = e_s^T M_p e_o$$

-   $e_s, e_o$ sono gli embedding di soggetto e oggetto
-   $M_p$ è la matrice della relazione $p$
-   Il punteggio è alto se la tripla è valida

### TransE

**TransE** rappresenta le relazioni come **traslazioni vettoriali**:

$$e_s + e_p \approx e_o$$

-   L'embedding del soggetto più l'embedding della relazione dovrebbe
    essere vicino all'embedding dell'oggetto
-   Il punteggio di una tripla è basato sulla distanza:

$$f(s, p, o) = -\|e_s + e_p - e_o\|$$

-   Semplice ed efficiente, ma limitato per relazioni complesse (es.
    1-a-molti)

``` mermaid
flowchart LR
    A[e_s] -->|+ e_p| B[e_o]
```

------------------------------------------------------------------------

## 📈 Applicazioni

  -----------------------------------------------------------------------
  Applicazione                         Descrizione
  ------------------------------------ ----------------------------------
  **Link Prediction**                  Predire relazioni mancanti tra
                                       entità

  **Entity Matching**                  Identificare se due entità sono la
                                       stessa

  **Fact Checking**                    Verificare la veridicità di
                                       affermazioni

  **Question Answering**               Rispondere a domande usando la
                                       conoscenza del grafo
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 📊 Riepilogo

  Modello      Idea                         Punteggio
  ------------ ---------------------------- ------------------------
  **RESCAL**   Relazioni come matrici       $e_s^T M_p e_o$
  **TransE**   Relazioni come traslazioni   $-\|e_s + e_p - e_o\|$

------------------------------------------------------------------------

```{=html}
<!-- ===== SEZIONE: 18_clustering.md ===== -->
```
# 18 --- Clustering Basato su Distanza

> *Riferimento: Elementi di Data Clustering --- apprendimento non
> supervisionato*

------------------------------------------------------------------------

## 🧩 Introduzione

Il **clustering** è una tecnica di **apprendimento non supervisionato**
che raggruppa i dati in **cluster** (gruppi) in modo che: - I punti
**nello stesso cluster** siano simili tra loro - I punti **in cluster
diversi** siano dissimili

Il clustering **basato su distanza** usa una **metrica di distanza** per
misurare la similarità tra i punti.

------------------------------------------------------------------------

## 📏 Metriche di Distanza

La scelta della metrica di distanza è fondamentale per il clustering
basato su distanza.

  ---------------------------------------------------------------------------------
  Metrica                     Formula                            Uso
  --------------------------- ---------------------------------- ------------------
  **Euclidea**                $\sqrt{\sum_i (x_i - y_i)^2}$      Spazi continui

  **Manhattan**               $\sum_i \lvert x_i - y_i \rvert$   Spazi a griglia

  **Chebyshev**               $\max_i \lvert x_i - y_i \rvert$   Massima differenza

  **Hamming**                 Numero di posizioni diverse        Dati
                                                                 simbolici/binari
  ---------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🎯 Algoritmi di Clustering

### K-Means

Il **K-Means** è l'algoritmo di clustering più comune:

1.  Scegli $K$ centroidi iniziali (casuali)
2.  **Assegnazione:** assegna ogni punto al centroide più vicino
3.  **Aggiornamento:** ricalcola i centroidi come media dei punti
    assegnati
4.  Ripeti finché i centroidi non cambiano

``` mermaid
flowchart LR
    A[Inizializza K centroidi] --> B[Assegna punti al centroide più vicino]
    B --> C[Ricalcola centroidi come media]
    C --> D{Convergenza?}
    D -->|No| B
    D -->|Sì| E[Cluster finali]
```

-   **Obiettivo:** minimizzare la somma delle distanze quadrate
    intra-cluster (within-cluster sum of squares)
-   **Sensibile** alla scelta iniziale dei centroidi e al numero $K$

### Gerarchico (Agglomerativo)

Il clustering **gerarchico agglomerativo** costruisce una gerarchia di
cluster: 1. Inizia con ogni punto come cluster singolo 2. Unisci
iterativamente i due cluster più vicini 3. Il risultato è un
**dendrogramma**

### DBSCAN

**DBSCAN** (Density-Based Spatial Clustering) raggruppa punti in base
alla **densità**: - I punti in regioni dense formano cluster - I punti
in regioni sparse sono **rumore/outlier** - Non richiede di specificare
il numero di cluster $K$

------------------------------------------------------------------------

## 📊 Riepilogo

  ---------------------------------------------------------------------------
  Algoritmo            Tipo           Vantaggi           Svantaggi
  -------------------- -------------- ------------------ --------------------
  **K-Means**          Partizionale   Semplice, veloce   Richiede $K$,
                                                         sensibile agli
                                                         outlier

  **Gerarchico**       Gerarchico     Produce            Costoso per dataset
                                      dendrogramma       grandi

  **DBSCAN**           Basato su      Gestisce rumore,   Sensibile ai
                       densità        non richiede $K$   parametri di densità
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

# 📎 Appendice --- Mappa dei capitoli

## PARTE I --- Fondamenti del Machine Learning

-   **01**

-   **02 --- Concetti Fondamentali** \## PARTE II --- Valutazione,
    metriche e strumenti

-   **03 --- Valutazione dei Modelli**

-   **04 --- Intervalli di Confidenza e Test Statistici**

-   **05 --- Predizione di Probabilità e Costi**

-   **06 --- Valutazione della Predizione Numerica**

-   **07 --- WEKA** \## PARTE III --- Modelli probabilistici e grafici

-   **08 --- Probabilistic Graphical Models (PGM)** \## PARTE IV ---
    Alberi, ensemble e kernel methods

-   **09 --- Alberi di Decisione e Regole**

-   **13 --- Boosting e Ensemble Learning**

-   **10 --- Kernel e Support Vector Machines (SVM)** \## PARTE V ---
    Deep Learning e modelli generativi

-   **11 --- Deep Learning**

-   **12 --- Tractable Probabilistic Models (TPM)** \## PARTE VI ---
    Natural Language Processing

-   **14 --- Natural Language Processing con Deep Learning** \## PARTE
    VII --- Reinforcement Learning

-   **15 --- Reinforcement Learning (RL)** \## PARTE VIII ---
    Apprendimento relazionale e rappresentazioni

-   **16 --- Apprendimento Relazionale e Inductive Logic Programming
    (ILP)**

-   **17 --- Knowledge Graph Embeddings (KGE)**

-   **18 --- Clustering Basato su Distanza**
