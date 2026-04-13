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
    echo "= 3 - Update all dependencies      ="
    echo "= -------------------------------- ="
    echo "= 0 - Exit                         ="
    echo "===================================="
    echo
}

install_module_dependencies() {
    local module_path=$1
    if [ -f "$module_path/requirements.txt" ]; then
        echo "Installing dependencies for module: $(basename $module_path)"
        pip install -r "$module_path/requirements.txt"
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to install dependencies for module $(basename $module_path)"
            return 1
        fi
        echo "✓ Module $(basename $module_path) dependencies installed"
    fi
    return 0
}

install_all_dependencies() {
    echo "Installing main dependencies from root requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Error installing main dependencies"
            return 1
        fi
        echo "✓ Main dependencies installed"
    else
        echo "Warning: root requirements.txt not found"
    fi
    
    echo
    echo "Checking for module dependencies..."
    
    if [ -d "modules" ]; then
        for module_dir in modules/*/; do
            if [ -d "$module_dir" ]; then
                install_module_dependencies "$module_dir"
            fi
        done
    else
        echo "No modules folder found, skipping module dependencies"
    fi
    
    return 0
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
    
    install_all_dependencies
    if [ $? -ne 0 ]; then
        echo "Error installing dependencies"
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

update_dependencies() {
    clear
    echo "Updating all dependencies..."
    echo
    
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found. Please run full installation first."
        read -p "Press Enter to continue..."
        return
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
    
    echo "Updating main dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install --upgrade -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "Error updating main dependencies"
        else
            echo "✓ Main dependencies updated"
        fi
    fi
    
    if [ -d "modules" ]; then
        echo
        echo "Updating module dependencies..."
        for module_dir in modules/*/; do
            if [ -d "$module_dir" ] && [ -f "$module_dir/requirements.txt" ]; then
                echo "Updating dependencies for module: $(basename $module_dir)"
                pip install --upgrade -r "$module_dir/requirements.txt"
                if [ $? -eq 0 ]; then
                    echo "✓ Module $(basename $module_dir) dependencies updated"
                fi
            fi
        done
    fi
    
    echo
    echo -e "\033[1;32mDependencies update completed!\033[0m"
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
    read -p "Select action [1-3 or 0]: " choice
    
    case $choice in
        1)
            install_bot
            ;;
        2)
            run_bot
            ;;
        3)
            update_dependencies
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