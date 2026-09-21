# Executive Crimson Design System — Technical Report Specification

This document extracts and formalizes the design language, typography, color palette, table architecture, and figure/diagram specifications used in the `@final-report` (`final_report.tex`). It serves as a comprehensive, reusable blueprint for creating publication-grade technical reports, capstone deliverables, and project documentation.

---

## 1. Design Philosophy & Aesthetic Core

The design follows an **Executive Crimson Academic** aesthetic:
- **Tone:** Authoritative, publication-grade, technical, and restrained.
- **Hierarchy:** Strong contrast between body text and structure; primary crimson branding frames headers and major dividers without overwhelming the page.
- **Density:** High information density with strict vertical rhythm; no loose or wasted whitespace.
- **Consistency:** Uniform table headers, caption typography, callout structures, and TikZ color tokens throughout the document.
- **Graceful Drafting:** Built-in safeguards such as image existence checks (`\shotscreenshot`) and section float barriers (`\FloatBarrier`) ensure error-free builds at any draft stage.

---

## 2. Page Geometry & Layout Foundations

The document is set on A4 paper with tight, balanced 2.0 cm margins to maximize printable area while keeping line lengths comfortable for dual-column and full-width reading.

```latex
\documentclass[11pt,a4paper]{article}

% --- Page Geometry ---
\usepackage[margin=2.0cm, top=2.0cm, bottom=2.0cm]{geometry}

% --- Paragraph Spacing & Formatting ---
\usepackage{parskip}
\setlength{\parskip}{3pt}      % Tight paragraph separation
\raggedbottom                  % Prevents awkward vertical stretching across pages
```

---

## 3. Typography & Font Hierarchy

The typography balances classic academic serif text with crisp modern sans-serif headings and structured monospace listings.

| Role | Font Package / Family | TeX Command / Scale | Semantic Usage |
| :--- | :--- | :--- | :--- |
| **Body & Math** | Times Roman (`mathptmx`) | `\usepackage{mathptmx}` | Body paragraphs, mathematical formulas, definitions. |
| **Headings & UI** | Helvetica (`helvet`) | `\usepackage[scaled=0.90]{helvet}` | Section headers, table header rows, diagram labels, captions. |
| **Monospace / Code** | TeX Typewriter (`ttfamily`) | Embedded default / fullflexible | Raw packet bytes, console commands, source code, filenames. |
| **Encoding & Micro** | `fontenc` (T1), `microtype` | `\usepackage[T1]{fontenc}`, `\usepackage{microtype}` | Kerning, protrusion, ligature hygiene across compilers. |

### Font Hierarchy Configuration

```latex
% Section formatting via titlesec
\usepackage{titlesec}

% Level 1 Section: Large, Bold Sans-Serif, Deep Crimson, with subtle bottom hairline rule
\titleformat{\section}
  {\sffamily\large\bfseries\color{accent}}
  {\thesection}{0.7em}{}[{\vspace{2pt}\color{rule}\titlerule[0.8pt]}]

% Level 2 Subsection: Medium, Bold Sans-Serif, Ink (Charcoal-Black)
\titleformat{\subsection}
  {\sffamily\normalsize\bfseries\color{ink}}
  {\thesubsection}{0.6em}{}

% Level 3 Subsubsection: Small Bold Sans-Serif
\titleformat{\subsubsection}
  {\sffamily\small\bfseries\color{ink}}
  {\thesubsubsection}{0.5em}{}

% Spacing around headings (left, before, after)
\titlespacing*{\section}{0pt}{16pt}{7pt}
\titlespacing*{\subsection}{0pt}{11pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{8pt}{3pt}
```

---

## 4. Color Palette & Visual Tokens

The color scheme is rooted in deep crimson and charcoal ink, supported by subtle tint washes and neutral framing greys.

### Color Swatches & Tokens

| Token Name | RGB Values | Hex Code | LaTeX Definition | Semantic Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `ink` | `(28, 32, 38)` | `#1C2026` | `\definecolor{ink}{RGB}{28,32,38}` | Primary text color. Soft near-black, less harsh than pure black. |
| `accent` / `primary` | `(155, 26, 32)` | `#9B1A20` | `\definecolor{accent}{RGB}{155,26,32}` | Deep Crimson. Primary brand color; used for section headers, caption titles, primary rules, and keypoint borders. |
| `accentstrong` | `(200, 16, 24)` | `#C81018` | `\definecolor{accentstrong}{RGB}{200,16,24}` | High-intensity red. Reserved for inline highlights (`\hl`), drops, alert badges, and security failures. |
| `accentsoft` | `(247, 228, 229)` | `#F7E4E5` | `\definecolor{accentsoft}{RGB}{247,228,229}` | Pale crimson tint. Used for table header backgrounds (`\hdrrow`), callout box fills, and highlighted diagram blocks. |
| `rule` | `(209, 213, 219)` | `#D1D5DB` | `\definecolor{rule}{RGB}{209,213,219}` | Hairline grey. Used for section dividers, table horizontal lines, code box borders, and flowchart outlines. |
| `codebg` | `(248, 246, 246)` | `#F8F6F6` | `\definecolor{codebg}{RGB}{248,246,246}` | Warm light-grey. Monospace code background, terminal box fill, diagram node background. |
| `hdrprimary` | `(155, 26, 32)` | `#9B1A20` | `\definecolor{hdrprimary}{RGB}{155,26,32}` | Header project title color. |
| `hdrsecondary` | `(175, 35, 42)` | `#AF232A` | `\definecolor{hdrsecondary}{RGB}{175,35,42}` | Header subtitle tint. |
| `darkslate` | `(33, 37, 41)` | `#212529` | `\definecolor{darkslate}{RGB}{33,37,41}` | Running header metadata and student ID text. |

### Color Definition Snippet

```latex
\usepackage{xcolor}
\definecolor{ink}{RGB}{28,32,38}          % Near-black body ink
\definecolor{accent}{RGB}{155,26,32}      % Deep crimson (primary accent)
\definecolor{primary}{RGB}{155,26,32}     % Alias used on cover page & rules
\definecolor{accentstrong}{RGB}{200,16,24}% Strong red for critical emphasis
\definecolor{accentsoft}{RGB}{247,228,229}% Pale crimson for table header fill
\definecolor{rule}{RGB}{209,213,219}      % Hairline grey for borders & lines
\definecolor{codebg}{RGB}{248,246,246}    % Subtle mono frame background
\definecolor{hdrprimary}{RGB}{155,26,32}  % Deep crimson for header
\definecolor{hdrsecondary}{RGB}{175,35,42}% Secondary crimson
\definecolor{darkslate}{RGB}{33, 37, 41}  % Charcoal metadata text
\color{ink}                               % Set default body text color
```

---

## 5. Header, Footer & Running Foliation

The running header establishes document context (project name, document type, course code, author IDs) on every body page, while the footer prints exact page counts ("Page X / Y") via `lastpage`.

```latex
\usepackage{fancyhdr}
\usepackage{lastpage}
\pagestyle{fancy}
\setlength{\headheight}{15pt}
\fancyhf{}

% --- Running Header ---
\fancyhead[L]{\small\textbf{\color{primary}PortScan-OS} \ $\cdot$ \ {\color{primary!85}Project Final Report}}
\fancyhead[R]{\small\color{darkslate!80}CSE 406 $\cdot$ 2105032 $\cdot$ 2105033}

% --- Running Footer ---
\fancyfoot[C]{\footnotesize\sffamily\color{ink}\thepage\ / \pageref{LastPage}}

% --- Top Rule: Thin Crimson Tinted Hairline ---
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\headrule}{%
  \vskip 2.5pt%
  {\color{primary!40}\hrule height \headrulewidth width \headwidth}%
  \vskip -\headrulewidth%
}
```

---

## 6. Cover Page / Titlepage Architecture

The cover page combines institutional dignity with bold visual framing:
1. **Institutional Emblem:** Scaled BUET / University logo (`0.22\textwidth`).
2. **Institution & Department:** Small-caps typography (`\textsc`).
3. **The Signature Crimson Title Block:** Framed between two 1mm thick `primary` rules.
4. **Subtitle & Document Type:** Clean, bold declaration.
5. **Team and Supervisor Columns:** Side-by-side balanced `minipage` blocks.
6. **Date:** Centered at the base.

```latex
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{0.4cm}

% --- LOGO ---
\includegraphics[width=0.22\textwidth]{rsc/buet.png}\\[0.8cm]

% --- INSTITUTION ---
{\Large \textsc{Bangladesh University of Engineering and Technology}}\\[0.3cm]
{\large \textsc{Department of Computer Science and Engineering}}\\[0.4cm]
{\normalsize CSE 406: Computer Security Lab \quad·\quad Semester January 2026}\\[1.6cm]

% --- TITLE BLOCK (Crimson Framed) ---
{\color{primary}\rule{\linewidth}{1mm}}\\[0.55cm]
{\LARGE\bfseries\color{primary} Port Scanning with OS Fingerprinting}\\[0.5cm]
{\large\bfseries Project 8: Attack Tools Implementation}\\[0.5cm]
{\color{primary}\rule{\linewidth}{1mm}}\\[1.4cm]

% --- SUBTITLE ---
{\Large\bfseries Final Report \& Implementation Demo}\\[2.6cm]

\vfill

% --- TEAM + SUPERVISOR SIDE-BY-SIDE ---
\begin{minipage}[t]{0.52\textwidth}
\centering
{\large\bfseries Submitted By}\\[0.5cm]
{\renewcommand{\arraystretch}{1.35}\normalsize
\begin{tabular}{rl}
\textbf{2105032} & Student Name 1\\
\textbf{2105033} & Student Name 2\\
\end{tabular}}
\end{minipage}%
\begin{minipage}[t]{0.48\textwidth}
\centering
{\large\bfseries Supervised By}\\[0.5cm]
{\large\bfseries Anwarul Bashir Shuaib}\\[0.15cm]
{\normalsize Lecturer}\\[0.05cm]
{\footnotesize Dept.\ of Computer Science \& Engineering, BUET}
\end{minipage}

\vspace{1.6cm}

% --- DATE ---
{\large \today}\\[0.4cm]
\end{titlepage}
```

---

## 7. Table Design & Specifications

Tables are a hallmark of the design: clean `booktabs` horizontal rules, full-width `tabularx` sizing, custom column types, and soft crimson header rows (`\hdrrow`).

### Table Design Rules
1. **Never use vertical rules:** No `|` characters in column format definitions.
2. **Width:** Always use `\begin{tabularx}{\textwidth}{...}` to span exact printable margins.
3. **Header Row:** Always start with `\hdrrow` (`\rowcolor{accentsoft}`) and bold sans-serif column text (`\thh{...}`).
4. **Column Alignment:**
   - Left-aligned with auto-hyphenation: `L{width}`.
   - Centered with fixed width: `C{width}`.
   - Flexible auto-wrapping content: `X`.
5. **Separation:** Use `\addlinespace[2pt]` or `\addlinespace[3pt]` between row categories for breathability.
6. **Subheadings:** Use `\multicolumn{N}{@{}l}{\textit{\color{accent}\textbf{Category Title}}}\\` to subdivide complex tables.
7. **Vertical Rhythm:** Set `\renewcommand{\arraystretch}{1.06}` to `1.25` inside the table environment.

### Helper Macros

```latex
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{colortbl}

% Custom column types
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}

% Table header formatting
\newcommand{\thh}[1]{\textbf{\sffamily #1}}
\newcommand{\hdrrow}{\rowcolor{accentsoft}}
```

### Table Template 1: Two-Column Feature / Specification Table

```latex
\begin{table}[!ht]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\caption{System Specification and Deliverable Breakdown}
\label{tab:system_specs}
\begin{tabularx}{\textwidth}{@{}L{4.2cm} X@{}}
\toprule
\hdrrow \thh{Component / Area} & \thh{Technical Description \& Operational Behavior}\\
\midrule
\multicolumn{2}{@{}l}{\textit{\color{accent}\textbf{Core Subsystem 1: Low-Level Engine}}}\\
Packet Framing & From-scratch byte packing with Python \texttt{struct}; RFC 1071 checksum computation.\\
Hardware Resolution & Windows IP Helper API (\texttt{iphlpapi.dll}) ARP pre-resolution via \texttt{SendARP}.\\
\addlinespace[2pt]
\multicolumn{2}{@{}l}{\textit{\color{accent}\textbf{Core Subsystem 2: Analysis Battery}}}\\
Heuristic Classification & Seven independent scoring strategies with probability damping below $\theta = 4.0$.\\
Active Probing & Diagnostic TCP option offering, ECN reflection, and rule-breaker flag testing.\\
\bottomrule
\end{tabularx}
\end{table}
```

### Table Template 2: Multi-Column Empirical Comparison Matrix

```latex
\begin{table}[!ht]
\centering
\scriptsize
\renewcommand{\arraystretch}{1.08}
\caption{Empirical Cross-Platform Evaluation Results}
\label{tab:eval_results}
\begin{tabularx}{\textwidth}{@{}L{2.8cm} L{1.8cm} L{2.2cm} X L{2.2cm}@{}}
\toprule
\hdrrow \thh{Target Environment} & \thh{Probe Mode} & \thh{Observed TTL / Win} & \thh{Key Characteristic Quirks} & \thh{Verdict \& Confidence}\\
\midrule
\textbf{Windows 11 Host}\newline\tiny(192.168.0.116:80) &
SYN/ACK Probe &
TTL 128 (0 hops)\newline Win 65535 &
M-W-S-T order; 1000\,Hz clock tick; strict incrementing IP-ID. &
\textbf{Windows}\newline(62\%, Tier 71\%)\\
\addlinespace[2pt]
\textbf{Kali Linux VM}\newline\tiny(192.168.56.101:80) &
Closed-Port RST &
TTL 64 (0 hops)\newline Win 0 (RST) &
\texttt{ALL\_ZERO} IP-ID (RFC 6864); ICMP Code 9 verbatim quote. &
\textbf{Linux}\newline(76\%, Tier 100\%)\\
\bottomrule
\end{tabularx}
\end{table}
```

---

## 8. Figure, Diagram & Screenshot Architecture

### 8.1 Captions
Captions use understated bold sans-serif crimson labels with small body text:
```latex
\usepackage{caption}
\captionsetup{
  labelfont={bf,sf,color=accent},
  textfont={small},
  labelsep=period,
  skip=6pt
}
```

### 8.2 Defensive Screenshot Handler (`\shotscreenshot`)
This macro solves a classic report authoring challenge: compiling a document before all screenshots have been taken. It verifies whether the target image file exists on disk. If found, it embeds it at the requested width; if missing, it renders an elegant bordered placeholder box showing the file path and target description.

```latex
% Usage: \shotscreenshot[optional width]{filepath}{caption text}
\newcommand{\shotscreenshot}[3][0.92\linewidth]{%
  \begin{center}
  \IfFileExists{#2}{%
    \includegraphics[width=#1]{#2}%
  }{%
    \setlength{\fboxrule}{0.6pt}\setlength{\fboxsep}{0pt}%
    \textcolor{rule}{\fbox{\parbox[c][2.6cm][c]{#1}{\centering
    \sffamily\footnotesize\color{accent}[\,DEMONSTRATION SCREENSHOT FRAME\,]\\[3pt]
    \color{ink}\normalfont\footnotesize #3\\[2pt]
    \tiny\ttfamily\color{ink!60}(File target: \detokenize{#2})}}}%
  }%
  \end{center}}
```

### 8.3 Multi-Panel Minipage Figures
To arrange subfigures cleanly without adding complex subcaption packages, use aligned `minipage` environments with explicit subheadings:

```latex
\begin{figure}[htbp]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
{\color{accent}\sffamily\bfseries\scriptsize (a) Component A Pipeline}\\[3pt]
% [TikZ or \includegraphics here]
\end{minipage}\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
{\color{accent}\sffamily\bfseries\scriptsize (b) Component B Pipeline}\\[3pt]
% [TikZ or \includegraphics here]
\end{minipage}
\caption{Comparative Architectural Overview of Primary and Secondary Subsystems.}
\label{fig:comparative_arch}
\end{figure}
```

### 8.4 Side-by-Side Text & Figure Layout (`\captionof`)
To place descriptive text directly beside a compact diagram:

```latex
\noindent
\begin{minipage}[t]{0.45\textwidth}
\vspace{0pt}
Paragraph text describing the network topology and architectural boundary conditions...
\end{minipage}\hfill
\begin{minipage}[t]{0.53\textwidth}
\vspace{0pt}
\centering
% [TikZ diagram here]
\vspace{-2pt}
\captionof{figure}{Heterogeneous verification testbed showing attacker and victim nodes.}
\label{fig:topology}
\end{minipage}
```

### 8.5 TikZ Diagram System & Styles

All diagrams share a cohesive aesthetic:
- **Font:** `font=\sffamily\scriptsize` or `\footnotesize`.
- **Arrows:** `Stealth[length=4pt]` or `Stealth[length=6pt]`.
- **Node Corner Radius:** Rounded corners (`2pt` to `4pt`).
- **Stroke Weights:** `0.6pt` (subtle/secondary), `0.8pt` (standard), `1.2pt` (accent/highlight).
- **Colors:** Nodes use `codebg`, borders use `rule` or `accent`, fills use `accentsoft`.

#### Standard Pipeline Flowchart Snippet
```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds}

\begin{figure}[htbp]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[
  font=\sffamily,
  stage/.style={draw=rule, fill=codebg, rounded corners=4pt, line width=0.8pt,
               text width=3.0cm, minimum height=1.8cm, align=center, inner sep=4pt},
  stage_acc/.style={stage, draw=accent, fill=accentsoft!60, line width=1.0pt},
  stage_hi/.style={stage, draw=accentstrong, fill=accentsoft, line width=1.2pt}
]
  \node[stage_acc] (s1) at (0, 0) {{\color{accent}\bfseries\scriptsize STEP 1}\\[2pt]\textbf{\small Ingestion}\\[2pt]\scriptsize Raw L2/L3 frames};
  \node[stage]     (s2) at (4.0, 0) {{\color{accent}\bfseries\scriptsize STEP 2}\\[2pt]\textbf{\small Processing}\\[2pt]\scriptsize Feature extraction};
  \node[stage_hi]  (s3) at (8.0, 0) {{\color{accentstrong}\bfseries\scriptsize STEP 3}\\[2pt]\textbf{\small Decision}\\[2pt]\scriptsize Confidence scoring};

  \draw[-{Stealth[length=6pt]}, line width=1.2pt, color=accent] (s1) -- (s2);
  \draw[-{Stealth[length=6pt]}, line width=1.2pt, color=accentstrong] (s2) -- (s3);
\end{tikzpicture}%
}
\caption{Three-Stage End-to-End Processing Architecture.}
\label{fig:pipeline}
\end{figure}
```

#### Protocol Ladder / Sequence Diagram Snippet
```latex
\begin{tikzpicture}[font=\sffamily\scriptsize]
    \node[draw=accent, fill=accentsoft, rounded corners=2pt, font=\bfseries, inner sep=2.5pt, minimum width=1.4cm] (c) at (0, 0) {Client};
    \node[draw=rule, fill=codebg, rounded corners=2pt, font=\bfseries, inner sep=2.5pt, minimum width=1.4cm] (s) at (3.0, 0) {Server};
    
    \draw[thick, color=accent] (c.south) -- (0, -2.8);
    \draw[thick, color=rule] (s.south) -- (3.0, -2.8);
    
    \draw[-{Stealth[length=4pt]}, thick] (0, -0.6) -- node[above, sloped, font=\tiny] {SYN ($seq = x$)} (3.0, -1.1);
    \draw[-{Stealth[length=4pt]}, thick, color=accent] (3.0, -1.4) -- node[above, sloped, font=\tiny] {SYN/ACK ($ack = x+1$)} (0, -1.9);
    \draw[-{Stealth[length=4pt]}, thick] (0, -2.2) -- node[above, sloped, font=\tiny] {ACK ($ack = y+1$)} (3.0, -2.7);
\end{tikzpicture}
```

---

## 9. Code Listings, Verbatim Boxes & Callouts

All callouts and terminal outputs use `tcolorbox` with sharp corners for a modern, architectural look.

### 9.1 Verbatim Console Log Box (`logbox`)
Uses `listings` integration so console outputs, special characters (`_`, `\`, `{`, `}`, `%`, `?`), and wide lines render cleanly without manual escaping:

```latex
\usepackage[most]{tcolorbox}
\tcbuselibrary{listings}

\newtcblisting{logbox}{
  breakable,
  enhanced,
  sharp corners,
  colback=codebg,
  colframe=rule,
  boxrule=0.5pt,
  left=8pt, right=8pt, top=6pt, bottom=6pt,
  listing only,
  listing options={
    basicstyle=\ttfamily\footnotesize\color{ink},
    breaklines=true,
    columns=fullflexible,
    keepspaces=true,
    aboveskip=0pt,
    belowskip=0pt
  }
}
```

### 9.2 Key-Point Callout Box (`keypoint`)
A restrained accent callout featuring a 2.5pt left border in `accent` with a pale `accentsoft` background:

```latex
\newtcolorbox{keypoint}[1][Key point]{
  enhanced,
  sharp corners,
  colback=accentsoft,
  colframe=accent,
  boxrule=0pt,
  leftrule=2.5pt,
  left=9pt, right=9pt, top=6pt, bottom=6pt,
  fonttitle=\sffamily\bfseries\footnotesize\color{accent},
  title={#1},
  coltitle=accent
}
```

### 9.3 Inline Highlight Macro (`\hl`)
Used sparingly to emphasize security states, verdicts, or failure causes:

```latex
\newcommand{\hl}[1]{\textbf{\color{accentstrong}#1}}
```

---

## 10. Front Matter & Document Progression

Reports follow a standardized flow:
1. **Titlepage:** Cover block, institution, team, supervisor, date.
2. **Executive Summary:** Unnumbered section (`\section*{...}`) with TOC line (`\addcontentsline{toc}{section}{...}`).
3. **Group Member Responsibilities Table:** Two-column assignment matrix.
4. **Table of Contents:** Linked in body ink (`{\hypersetup{linkcolor=ink}\tableofcontents}`).
5. **Body Sections:** Modular files via `\input{sections/...}`.
6. **Section Float Barriers:** `\usepackage[section]{placeins}` and explicit `\FloatBarrier` calls prevent floats from spilling across major topical boundaries.

---

## 11. Minimal Working Example (MWE) / Reusable Starter Template

Save the following code as `report_template.tex` to instantly generate a document using this design system:

```latex
\documentclass[11pt,a4paper]{article}

% ==============================================================================
% 1. GEOMETRY & TYPOGRAPHY
% ==============================================================================
\usepackage[margin=2.0cm, top=2.0cm, bottom=2.0cm]{geometry}
\usepackage{mathptmx}                 % Times body + math
\usepackage[scaled=0.90]{helvet}      % Helvetica for headings/sans
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{parskip}
\setlength{\parskip}{3pt}
\raggedbottom

% ==============================================================================
% 2. TABLES, LISTS & FLOATS
% ==============================================================================
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{colortbl}
\usepackage{enumitem}
\setlist{nosep, leftmargin=1.4em}
\usepackage{float}
\usepackage{graphicx}
\usepackage{caption}
\usepackage[section]{placeins}

\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcommand{\thh}[1]{\textbf{\sffamily #1}}
\newcommand{\hdrrow}{\rowcolor{accentsoft}}

% ==============================================================================
% 3. COLOR PALETTE
% ==============================================================================
\usepackage{xcolor}
\definecolor{ink}{RGB}{28,32,38}          % Near-black body ink (#1C2026)
\definecolor{accent}{RGB}{155,26,32}      % Deep crimson accent (#9B1A20)
\definecolor{primary}{RGB}{155,26,32}     % Alias for primary brand
\definecolor{accentstrong}{RGB}{200,16,24}% Vivid red emphasis (#C81018)
\definecolor{accentsoft}{RGB}{247,228,229}% Pale crimson fill (#F7E4E5)
\definecolor{rule}{RGB}{209,213,219}      % Hairline grey (#D1D5DB)
\definecolor{codebg}{RGB}{248,246,246}    % Code & terminal background (#F8F6F6)
\definecolor{hdrprimary}{RGB}{155,26,32}  % Header brand color
\definecolor{darkslate}{RGB}{33, 37, 41}  % Charcoal metadata (#212529)
\color{ink}

% ==============================================================================
% 4. HEADINGS & CAPTIONS
% ==============================================================================
\usepackage{titlesec}
\titleformat{\section}
  {\sffamily\large\bfseries\color{accent}}
  {\thesection}{0.7em}{}[{\vspace{2pt}\color{rule}\titlerule[0.8pt]}]
\titleformat{\subsection}
  {\sffamily\normalsize\bfseries\color{ink}}
  {\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{16pt}{7pt}
\titlespacing*{\subsection}{0pt}{11pt}{4pt}

\captionsetup{labelfont={bf,sf,color=accent}, textfont={small}, labelsep=period, skip=6pt}

% ==============================================================================
% 5. HEADERS & FOOTERS
% ==============================================================================
\usepackage{fancyhdr}
\usepackage{lastpage}
\pagestyle{fancy}
\setlength{\headheight}{15pt}
\fancyhf{}
\fancyhead[L]{\small\textbf{\color{primary}MyProject-Title} \ $\cdot$ \ {\color{primary!85}Technical Report}}
\fancyhead[R]{\small\color{darkslate!80}CSE 4XX $\cdot$ StudentID}
\fancyfoot[C]{\footnotesize\sffamily\color{ink}\thepage\ / \pageref{LastPage}}
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\headrule}{%
  \vskip 2.5pt%
  {\color{primary!40}\hrule height \headrulewidth width \headwidth}%
  \vskip -\headrulewidth%
}

% ==============================================================================
% 6. TCOLORBOX & HELPERS
% ==============================================================================
\usepackage[most]{tcolorbox}
\tcbuselibrary{listings}

\newtcblisting{logbox}{
  breakable, enhanced, sharp corners,
  colback=codebg, colframe=rule, boxrule=0.5pt,
  left=8pt, right=8pt, top=6pt, bottom=6pt,
  listing only,
  listing options={
    basicstyle=\ttfamily\footnotesize\color{ink},
    breaklines=true, columns=fullflexible, keepspaces=true,
    aboveskip=0pt, belowskip=0pt}}

\newtcolorbox{keypoint}[1][Key point]{
  enhanced, sharp corners,
  colback=accentsoft, colframe=accent, boxrule=0pt,
  leftrule=2.5pt, left=9pt, right=9pt, top=6pt, bottom=6pt,
  fonttitle=\sffamily\bfseries\footnotesize\color{accent},
  title={#1}, coltitle=accent}

\newcommand{\shotscreenshot}[3][0.92\linewidth]{%
  \begin{center}
  \IfFileExists{#2}{%
    \includegraphics[width=#1]{#2}%
  }{%
    \setlength{\fboxrule}{0.6pt}\setlength{\fboxsep}{0pt}%
    \textcolor{rule}{\fbox{\parbox[c][2.6cm][c]{#1}{\centering
    \sffamily\footnotesize\color{accent}[\,DEMONSTRATION SCREENSHOT FRAME\,]\\[3pt]
    \color{ink}\normalfont\footnotesize #3\\[2pt]
    \tiny\ttfamily\color{ink!60}(File target: \detokenize{#2})}}}%
  }%
  \end{center}}

\newcommand{\hl}[1]{\textbf{\color{accentstrong}#1}}

\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}
\usepackage[hidelinks]{hyperref}

% ==============================================================================
% DOCUMENT BODY
% ==============================================================================
\begin{document}

% --- COVER PAGE ---
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{0.5cm}
{\Large \textsc{Bangladesh University of Engineering and Technology}}\\[0.3cm]
{\large \textsc{Department of Computer Science and Engineering}}\\[0.4cm]
{\normalsize Course Code: Course Name \quad·\quad Semester Year}\\[2.0cm]

{\color{primary}\rule{\linewidth}{1mm}}\\[0.55cm]
{\LARGE\bfseries\color{primary} Project Main Title Here}\\[0.5cm]
{\large\bfseries Subtitle or Track Designation}\\[0.5cm]
{\color{primary}\rule{\linewidth}{1mm}}\\[2.5cm]

\vfill

\begin{minipage}[t]{0.48\textwidth}
\centering
{\large\bfseries Submitted By}\\[0.4cm]
\textbf{21050XX} & Author Name
\end{minipage}%
\begin{minipage}[t]{0.48\textwidth}
\centering
{\large\bfseries Supervised By}\\[0.4cm]
Supervisor Name\\[0.1cm]
{\footnotesize Department Name, Institution}
\end{minipage}

\vspace{1.8cm}
{\large \today}
\end{titlepage}

% --- FRONT MATTER ---
\section*{\sffamily\color{accent}Executive Summary}
\addcontentsline{toc}{section}{Executive Summary}
This document outlines the system architecture and verification results...

\clearpage
{\hypersetup{linkcolor=ink}\tableofcontents}
\clearpage

% --- MAIN BODY ---
\section{System Architecture}
Here is an overview of the core architectural components.

\begin{table}[htbp]
\centering
\small
\renewcommand{\arraystretch}{1.15}
\caption{System Module Inventory}
\label{tab:modules}
\begin{tabularx}{\textwidth}{@{}L{3.5cm} X@{}}
\toprule
\hdrrow \thh{Module} & \thh{Description}\\
\midrule
Core Engine & Processes raw frames and parses RFC 791 headers.\\
Verification Suite & Automated end-to-end regression validation.\\
\bottomrule
\end{tabularx}
\end{table}

\begin{keypoint}[Critical Design Consideration]
The system employs strict input isolation to prevent unexpected state degradation.
\end{keypoint}

\begin{logbox}
[*] Initializing scan on 192.168.1.1 ...
[+] Found open port: 80/tcp (HTTP)
[+] Verification passed with 0 errors.
\end{logbox}

\end{document}
```
