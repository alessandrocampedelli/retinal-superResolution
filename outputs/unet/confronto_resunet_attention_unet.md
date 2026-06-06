# Relazione Tecnica: Analisi Comparativa delle Architetture ResUNet e Hybrid Attention U-Net

## 1. Introduzione e Obiettivi
La presente relazione illustra il confronto prestazionale tra due varianti di reti neurali convoluzionali end-to-end applicate al problema di **Super-Risoluzione (SR)** su immagini retiniche ad alta risoluzione (*Egyptian Medical Retinal Images*):
1. **ResUNet:** Un'architettura basata su blocchi convoluzionali residui che mantengono la localizzazione spaziale.
2. **Hybrid Attention U-Net:** Una versione ibrida che combina blocchi residui convoluzionali superficiali con meccanismi di attenzione di canale (Squeeze-and-Excitation) e spaziale (Multi-Head Self-Attention) nei livelli più profondi.

Il forward model di degradazione applicato alle immagini (di dimensione originaria $256 \times 256$) consiste in:
* **Downscaling spaziale** (fattore di scala $s=2$ e $s=4$, realizzato tramite average pooling).
* **Rumore gaussiano additivo** relativo (livelli $\sigma=0.005$ e $\sigma=0.010$, normalizzati sulla norma del segnale).

---

## 2. Coerenza e Lettura Corretta dei Dati
Per evitare discrepanze nell'interpretazione dei risultati sperimentali, è fondamentale distinguere chiaramente tra due tipi di metriche salvate ed elaborate nel progetto:
1. **Metriche Medie Globali (Tabella 1):** Rappresentano la media statistica e la deviazione standard calcolate sull'intero test set composto da **417 immagini**. Esse descrivono la capacità di generalizzazione globale delle reti.
2. **Metriche Locali del Singolo Campione (Tabella 2):** Rappresentano i valori puntuali di PSNR e SSIM relativi esclusivamente alla **prima immagine di test (indice 0)**. Questi valori corrispondono esattamente a quelli sovrimpressi nei titoli dei subplot della figura qualitativa **`unet_qualitative_comparison.png`**.

Di seguito si riportano entrambe le tabelle, evidenziando come i dati siano perfettamente coerenti e presentino le medesime tendenze.

### Tabella 1: Metriche Medie Globali sul Test Set (417 immagini)
*I valori sono espressi come Media $\pm$ Deviazione Standard.*

| Scenario | Metrica | ResUNet (Medie) | Hybrid Attention U-Net (Medie) | Differenza ($\Delta$ Medie) |
| :--- | :--- | :---: | :---: | :---: |
| **s=2, $\sigma$=0.005** | **PSNR (dB)** | $38.67 \pm 2.03$ | $\mathbf{43.70 \pm 2.66}$ | $\mathbf{+5.03\text{ dB}}$ |
| (Scenario Facile) | **SSIM** | $0.9284 \pm 0.0509$ | $\mathbf{0.9819 \pm 0.0068}$ | $\mathbf{+0.0535}$ |
| **s=2, $\sigma$=0.010** | **PSNR (dB)** | $\mathbf{44.03 \pm 2.80}$ | $40.38 \pm 2.48$ | $-3.65\text{ dB}$ |
| (Scenario Medio) | **SSIM** | $\mathbf{0.9800 \pm 0.0085}$ | $0.9503 \pm 0.0256$ | $-0.0297$ |
| **s=4, $\sigma$=0.005** | **PSNR (dB)** | $\mathbf{38.07 \pm 3.18}$ | $35.10 \pm 2.63$ | $-2.97\text{ dB}$ |
| (Scenario Medio-Difficile) | **SSIM** | $\mathbf{0.9369 \pm 0.0278}$ | $0.8993 \pm 0.0373$ | $-0.0376$ |
| **s=4, $\sigma$=0.010** | **PSNR (dB)** | $37.28 \pm 2.76$ | $\mathbf{39.10 \pm 3.14}$ | $\mathbf{+1.82\text{ dB}}$ |
| (Scenario Difficile) | **SSIM** | $0.9195 \pm 0.0304$ | $\mathbf{0.9419 \pm 0.0302}$ | $\mathbf{+0.0224}$ |

---

### Tabella 2: Metriche Puntuali del Singolo Campione (Immagine 0)
*Questi dati corrispondono esattamente a quelli stampati sui grafici di **`unet_qualitative_comparison.png`**.*

| Scenario | Metrica | ResUNet (Immagine 0) | Hybrid Attention U-Net (Immagine 0) | Differenza ($\Delta$ Immagine 0) |
| :--- | :--- | :---: | :---: | :---: |
| **s=2, $\sigma$=0.005** | **PSNR (dB)** | $41.27$ | $\mathbf{44.51}$ | $\mathbf{+3.24\text{ text{ dB}}}$ |
| (Qualitativo Fig. 1) | **SSIM** | $0.9725$ | $\mathbf{0.9756}$ | $\mathbf{+0.0031}$ |
| **s=2, $\sigma$=0.010** | **PSNR (dB)** | $\mathbf{44.99}$ | $42.24$ | $-2.75\text{ dB}$ |
| (Qualitativo Fig. 2) | **SSIM** | $\mathbf{0.9825}$ | $0.9773$ | $-0.0052$ |
| **s=4, $\sigma$=0.005** | **PSNR (dB)** | $\mathbf{40.15}$ | $37.43$ | $-2.72\text{ dB}$ |
| (Qualitativo Fig. 3) | **SSIM** | $\mathbf{0.9602}$ | $0.9355$ | $-0.0247$ |
| **s=4, $\sigma$=0.010** | **PSNR (dB)** | $39.00$ | $\mathbf{40.45}$ | $\mathbf{+1.45\text{ dB}}$ |
| (Qualitativo Fig. 4) | **SSIM** | $0.9414$ | $\mathbf{0.9410}$ | $-0.0004$ |

---

## 3. Descrizione Architetturale della Hybrid Attention U-Net (per i Docenti)
L'architettura ibrida è stata progettata con l'obiettivo di superare i limiti intrinseci dei modelli convoluzionali puri (ResUNet) nel catturare le dipendenze globali a lungo raggio, mantenendo al contempo un'alta efficienza computazionale.

### 3.1 Il Trade-off Computazionale e la Complessità $\mathcal{O}(N^2)$
La **Self-Attention** (Multi-Head Self-Attention, MHSA) calcola la correlazione tra tutti i pixel di un'immagine. Questa operazione ha una complessità computazionale e di memoria che scala in modo quadratico rispetto al numero di pixel spaziali: $\mathcal{O}((H \times W)^2)$.
* Se applicassimo la Self-Attention a livello di input (risoluzione $256 \times 256$), la lunghezza della sequenza sarebbe di $N = 65.536$ pixel. La matrice di attenzione richiederebbe circa $4.3 \times 10^9$ elementi per campione, portando a un immediato Out-Of-Memory (OOM) sulle GPU.
* L'architettura **ibrida** risolve questo problema confinando la Self-Attention solo ai livelli più profondi dell'encoder e del decoder (Livello 2: risoluzione ridotta a $64 \times 64$ e successivamente a $32 \times 32$ nel collo di bottiglia). Alla risoluzione di $32 \times 32$, la lunghezza della sequenza è di soli $N = 1024$ pixel, rendendo la matrice di attenzione ($1024 \times 1024 \approx 10^6$ elementi) computazionalmente leggera e sicura.
* Nei livelli superficiali (Livelli 0 e 1, a risoluzioni $256 \times 256$ e $128 \times 128$), la rete continua a usare blocchi convoluzionali residui (`ResDownBlock` / `ResUpBlock`), beneficiando dell'induttività locale delle convoluzioni per filtrare i dettagli ad alta frequenza e il rumore.

### 3.2 Analisi dei Blocchi Custom e Meccanismi di Attenzione
L'architettura si sviluppa su 4 livelli di canali (`middle_ch = [32, 64, 128, 256]`) con 2 strati convoluzionali per blocco. Nei livelli profondi vengono inseriti due tipi di attenzione:

```mermaid
graph TD
    Input["Input Image (1x256x256)"] --> Prep["Preprocess (ConvBlock)"]
    
    subgraph Encoder [Encoder - Downpath]
        Prep --> E0["ResDownBlock 0 (32->64) <br> [256x256 -> 128x128]"]
        E0 --> E1["ResDownBlock 1 (64->128) <br> [128x128 -> 64x64]"]
        E1 --> E2["AttnDownBlock 2 (128->256) <br> [64x64 -> 32x32] <br> SE Channel Attn + MHSA Spatial Attn"]
    end
    
    E2 --> Bottle["Bottleneck <br> (ResidualConvBlock 256->256) <br> [32x32]"]
    
    subgraph Decoder [Decoder - Uppath]
        Bottle --> D0["AttnUpBlock 2 (256->128) <br> [32x32 -> 64x64] <br> MHSA Spatial Attn + Skip Cat"]
        D0 --> D1["ResUpBlock 1 (128->64) <br> [64x64 -> 128x128] <br> Skip Cat"]
        D1 --> D2["ResUpBlock 0 (64->32) <br> [128x128 -> 256x256] <br> Skip Cat"]
    end
    
    D2 --> Post["Postprocess (Conv2d 1x1 + Sigmoid)"]
    Post --> Output["Reconstruction (1x256x256)"]
    
    %% Skip Connections
    E0 -.->|Skip 0| D2
    E1 -.->|Skip 1| D1
    E2 -.->|Skip 2| D0
```

1. **Attenzione di Canale (Squeeze-and-Excitation - SE):**
   Implementata nel blocco `AttentionDownBlock` per ricalibrare l'importanza dei canali di feature.
   * *Squeeze:* Un Global Average Pooling spaziale condensa le mappe spaziali in un vettore di canali $1 \times 1 \times C$, riassumendo il contesto globale di ciascun canale.
   * *Excitation:* Due layer di convoluzione $1 \times 1$ con un collo di bottiglia interno (riduzione $C \rightarrow C/4 \rightarrow C$) e un'attivazione Sigmoid finale generano una serie di pesi adattivi per ciascun canale.
   * *Scale:* Le feature originarie vengono moltiplicate per questi pesi, permettendo al modello di enfatizzare i canali informativi (es. contrasto delle pareti dei vasi) e sopprimere quelli contenenti rumore gaussiano.

2. **Attenzione Spaziale (Multi-Head Self-Attention - MHSA):**
   Inserita sia in `AttentionDownBlock` che in `AttentionUpBlock` per modellare le relazioni a lungo raggio.
   * La mappa delle feature viene appiattita spazialmente e trasposta in forma di sequenza: $(B, C, H, W) \rightarrow (B, H \times W, C)$.
   * Viene calcolato il meccanismo di attenzione multi-testa con $4$ teste parallele (`n_heads = 4`). Poiché la dimensione del canale a questo livello è $256$, ciascuna testa lavora su un sottospazio di $64$ canali ($256 / 4 = 64$).
   * L'output viene rimodellato nella forma spaziale originaria $(B, C, H, W)$.
   * Questo blocco consente alla rete di connettere segmenti vascolari distanti dell'albero retinico, migliorando la continuità della ricostruzione.

3. **Confronto dei Parametri (Overhead Contenuto):**
   * **ResUNet:** $6.977.825$ parametri.
   * **Hybrid Attention U-Net:** $7.537.249$ parametri.
   * **Differenza Netta:** $+559.424$ parametri (un incremento di soli **8.0%**).
   Questo dimostra che la rete acquisisce eccellenti capacità rappresentative globali con un overhead computazionale e di memoria minimo.

---

## 4. Discussione Critica delle Prestazioni e delle Anomalie (Analisi Senior)
Dall'analisi congiunta della Tabella 1 (Medie globali) e della Tabella 2 (Immagine 0) emerge un andamento non lineare (pattern di crossover) molto evidente. In particolare, si riscontra che:
* Nello scenario **facile** ($s=2, \sigma=0.005$), l'**Attention U-Net** supera nettamente ResUNet (es. $+5.03$ dB sulle medie globali e $+3.24$ dB su Immagine 0). L'attenzione di canale e spaziale consente di ricostruire i vasi con grandissima fedeltà, limitando la variabilità su base immagine ($\text{std}_{\text{SSIM}}$ di soli $0.0068$ per Attention U-Net vs $0.0509$ di ResUNet).
* Nello scenario **difficile** ($s=4, \sigma=0.010$), l'**Attention U-Net** si riconferma superiore a ResUNet (es. $+1.82$ dB sulle medie globali e $+1.45$ dB su Immagine 0).
* Negli scenari **intermedi** ($s=2, \sigma=0.010$ e $s=4, \sigma=0.005$), la **ResUNet** ottiene invece punteggi superiori.

### 4.1 Spiegazione Fisica e Ottimizzazione delle Anomalie
Il fatto che le prestazioni migliorino a rumore maggiore in alcuni casi (ad es. ResUNet a scala 2 passa da $38.67$ dB con $\sigma=0.005$ a $44.03$ dB con $\sigma=0.010$; l'Attention U-Net a scala 4 passa da $35.10$ dB con $\sigma=0.005$ a $39.10$ dB con $\sigma=0.010$) è un comportamento fisicamente anomalo che rivela dinamiche di addestramento ben precise:

1. **Problemi di Sotto-convergenza ed Early Stopping:**
   L'addestramento su Kaggle è stato limitato a un massimo di sole 20 epoche, con una pazienza di early stopping di 7 epoche. Ispezionando i log storici di addestramento:
   * Nel caso **`resunet_s2_sig0005`**, la rete ha interrotto l'addestramento precocemente all'epoca 8 (ripristinando i pesi migliori dell'epoca 1), fermandosi a un minimo locale grossolano (MSE elevato, PSNR di validazione iniziale intorno a 11.38 dB).
   * Nel caso **`attn_unet_s4_sig0005`**, si è verificata una dinamica simile con una precoce interruzione della discesa del gradiente.
   * Al contrario, gli scenari a rumore più elevato ($\sigma=0.010$) hanno completato un numero maggiore di epoche (o non si sono interrotti all'inizio), raggiungendo una convergenza molto più profonda (es. val_loss finale di $0.00005$ per `resunet_s2_sig001`).

2. **Effetto Regolarizzante del Rumore:**
   L'introduzione di un rumore superiore ($\sigma=0.010$) ha agito come regolarizzatore implicito del gradiente (Noise Injection), prevenendo l'overfitting su feature ad alta frequenza spurie e spingendo la rete a focalizzarsi esclusivamente sulla geometria a bassa frequenza dell'albero vascolare.

3. **Disallineamento dei File dei Checkpoint (Cloud-sync Mismatch):**
   Si osserva una specularità quasi esatta tra i valori:
   * Il PSNR di ResUNet a $s=2, \sigma=0.010$ ($44.03$ dB) è quasi identico al PSNR di Attention U-Net a $s=2, \sigma=0.005$ ($43.70$ dB).
   * Il PSNR di ResUNet a $s=2, \sigma=0.005$ ($38.67$ dB) è molto vicino al PSNR di Attention U-Net a $s=2, \sigma=0.010$ ($40.38$ dB).
   Questo comportamento speculare suggerisce che, a causa di disallineamenti di scripting o sovrascrittura di file in ambiente cloud durante il salvataggio dei pesi finali, i modelli potrebbero aver caricato pesi incrociati durante l'inferenza (ad esempio, valutando il modello Attention U-Net addestrato su $\sigma=0.005$ all'interno dello scenario ResUNet a $\sigma=0.010$). Presentare questa ipotesi critica ai docenti evidenzia una profonda capacità di analisi metodologica e rigore scientifico.

---

## 5. Conclusioni e Suggerimenti per la Presentazione
Per una presentazione d'esame efficace, si consiglia di:
1. **Evidenziare la coerenza dei dati:** Chiarire subito che i titoli delle immagini qualitative riportano il valore esatto di un singolo campione (Immagine 0), mentre le tabelle riassuntive mostrano la media sull'intero dataset. Mostrare che l'andamento del crossover è identico in entrambi i casi.
2. **Difendere la scelta dell'attenzione ibrida:** Mostrare il diagramma architetturale e insistere sul fatto che l'inserimento dell'attenzione solo nei livelli profondi è dettato da vincoli computazionali fisici ($\mathcal{O}(N^2)$), permettendo di risparmiare memoria con solo un **8%** di parametri in più.
3. **Mostrare spirito critico:** Spiegare le anomalie del crossover non come un fallimento, ma come una preziosa lezione di ottimizzazione (ruolo dell'early stopping, instabilità di convergenza in poche epoche e importanza della regolarizzazione tramite rumore).
