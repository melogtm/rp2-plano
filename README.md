# Plano de Trabalho — Resolução de Problemas II (ACH0042)

Repositório do trabalho em grupo de RP II, 2026/2 (EACH-USP).

O artigo é escrito em LaTeX. Cada um edita localmente e commita; o GitHub Actions compila o PDF a cada push. **Ninguém precisa instalar LaTeX pra contribuir** — dá pra escrever a sua seção e pegar o PDF pronto no CI. Mas instalar dá feedback em segundos em vez de minutos, então vale.

## Estrutura

```
.
├── .github/workflows/build.yml   # CI: compila e publica o PDF
├── .vscode/settings.json         # config de build (versionada, não mexer)
├── main.tex                      # o artigo — arquivo único, de propósito
├── referencias.bib               # bibliografia
├── figuras/
└── .gitignore
```

Arquivo único em vez de um `.tex` por seção: com três pessoas commitando, fragmentar troca conflito de merge por conflito de merge **mais** erro de compilação em arquivo que ninguém abriu. Pra 4–6 páginas não compensa.

## Instalação

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install texlive-latex-recommended texlive-latex-extra \
  texlive-lang-portuguese texlive-fonts-recommended latexmk
```

### macOS

```bash
brew install --cask basictex
```

BasicTeX tem ~100 MB, mas vem pelado. Complete com o que o template precisa:

```bash
eval "$(/usr/libexec/path_helper)"      # coloca o tlmgr no PATH desta sessão
sudo tlmgr update --self
sudo tlmgr install latexmk babel-portuges collection-fontsrecommended
```

Se preferir não caçar pacote faltando um a um, use o MacTeX completo (`brew install --cask mactex`) — ~5 GB, mas resolve tudo de uma vez.

### Verificar

```bash
latexmk -pdf main.tex
```

Gerou `main.pdf`? Está pronto.

## Editor

VS Code + extensão **LaTeX Workshop** (`James-Yu.latex-workshop`).

A config já está em `.vscode/settings.json`, então ao clonar você herda tudo. Na primeira compilação (`Ctrl+Alt+B` / `Cmd+Alt+B`), escolha a receita **latexmk (latexmk)**.

- `Ctrl+Alt+V` — abre o preview do PDF lado a lado
- Salvar o arquivo já dispara a compilação
- Clicar no PDF pula pra linha correspondente no `.tex` (SyncTeX)

Use `latexmk`, não `pdflatex` sozinho: o `latexmk` roda a sequência pdflatex → bibtex → pdflatex quantas vezes for necessário. Com `pdflatex` puro, as citações saem como `[?]`.

## Build no CI

Todo push na `main` e todo pull request disparam o workflow. Pra baixar o PDF:

**Actions** → clica no run → **Artifacts** (rodapé) → `plano-rp2`

Artifacts ficam disponíveis por 90 dias.

O CI roda em PRs justamente pra quebrar antes de entrar na `main`. Se o build falhar no PR, o problema é seu; se falhar na `main`, é problema de todo mundo.

## Fluxo de trabalho

1. `git pull` antes de começar a escrever — sempre
2. Trabalhe na sua seção; evite reformatar linhas de outra pessoa (vira conflito por nada)
3. Compile local antes de commitar — não empurre `.tex` quebrado
4. Commits pequenos e frequentes; mensagem dizendo a seção (`metodo: primeira versão`)

Se for por branch: uma branch por seção, PR pra `main`, merge quando o CI passar.

## Entregas

| Data | Entrega | Peso |
|---|---|---|
| 26/08 | Plano de trabalho (P1) | 5% |
| 16/09 | Avanços do relatório | — |
| 21/10 | Relatório parcial + vídeo 8–10 min (P2) | 5% |
| 18/11 | Relatório final (P3) | 90% |
| 19/11 | Slides da apresentação | — |

Só um integrante submete cada entrega. Todos apresentam no vídeo e na apresentação final.

P1 e P2 valem só 10% somados, mas entram na fórmula como multiplicador: não entregar zera a média final, independente do P3.

## Pendências

- [ ] Substituir o template placeholder pelo oficial do e-Disciplinas