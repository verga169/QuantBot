#!/usr/bin/env bash
# ==============================================================================
# BYBIT QUANT BOT - AVVIO APPLICAZIONE TERMINALE LINUX MINT
# ==============================================================================

# Risolve la directory corrente dello script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Controlla l'esistenza del virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "Virtual environment .venv non trovato! Lo creo..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Imposta il titolo della finestra esclusivamente per questo terminale
printf '\033]0;Quant Bot\007' 2>/dev/null || true

# Imposta la dimensione della finestra terminale (80 righe x 257 colonne)
printf '\e[8;80;257t' 2>/dev/null || true

# Avvia l'interfaccia terminale TUI con riavvio automatico se non uscita esplicita
while true; do
    python app_tui.py
    code=$?
    if [ $code -eq 0 ]; then
        break
    fi
    echo "Riavvio interfaccia TUI in corso..."
    sleep 1
done

# ==============================================================================
