#!/bin/bash

export LANG=ru_RU.UTF-8
export LC_ALL=ru_RU.UTF-8

cd "$(dirname "$0")"

show_menu() {
    clear
    echo -e "\033[1;36m"
    echo "####################################"
    echo "#      NeuroVT - Control Menu      #"
    echo "####################################"
    echo -e "\033[0m"
    echo
    echo "===================================="
    echo "= 1 - Full installation            ="
    echo "= 2 - Run without installation     ="
    echo "= -------------------------------- ="
    echo "= 0 - Exit                         ="
    echo "===================================="
    echo
}

install_bot() {
    clear
    echo "Starting automatic installation..."
    echo "This may take several minutes depending on your internet connection..."
    echo
    
    echo "Starting installation process..."
    
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            echo "Error creating virtual environment"
            read -p "Press Enter to continue..."
            return
        fi
    else
        echo "Virtual environment already exists, skipping creation..."
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "Error activating virtual environment"
        read -p "Press Enter to continue..."
        return
    fi
    
    echo "Updating pip..."
    pip install --upgrade pip
    if [ $? -ne 0 ]; then
        echo "Error updating pip"
        read -p "Press Enter to continue..."
        return
    fi
    
    echo "Installing main dependencies..."
    pip install torchaudio
    pip install flask
    pip install torch
    pip install omegaconf
    pip install torchcodec
    pip install numpy
    pip install soundfile
    pip install openai
    pip install vosk
    pip install pyaudio
    if [ $? -ne 0 ]; then
        echo "Error installing main dependencies"
        read -p "Press Enter to continue..."
        return
    fi
    
    sleep 3
    clear
    
    echo -e "\033[1;32m"
    echo "##################################################################"
    echo "#              Installation completed successfully!              #"
    echo "##################################################################"
    echo "=                                                                ="
    echo "=       You can now run the application from the main menu       ="
    echo "=                                                                ="
    echo "=================================================================="
    echo -e "\033[0m"
    echo
    
    sleep 3
    clear
    
    source venv/bin/activate
    python3 main.py
    
    read -p "Press Enter to continue..."
}

run_bot() {
    clear
    echo "Starting main.py..."
    
    has_error=0
    
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to activate virtual environment"
            has_error=1
        fi
    else
        echo "Warning: Virtual environment not found"
        has_error=1
    fi
    
    echo -e "\033[1;36m"
    python3 main.py
    if [ $? -ne 0 ]; then
        has_error=1
    fi
    echo -e "\033[0m"
    
    read -p "Press Enter to continue..."
}

while true; do
    show_menu
    read -p "Select action [1-0]: " choice
    
    case $choice in
        1)
            install_bot
            ;;
        2)
            run_bot
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Press Enter to continue..."
            read
            ;;
    esac
done