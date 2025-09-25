# YALM Generator 🚀

[![Estado del Build](https://img-shields.io/badge/build-passing-brightgreen?style=for-the-badge)](https://github.com)
[![Licencia: MIT](https://img-shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Versión de R](https://img-shields.io/badge/R-%3E%3D%204.0-blue?style=for-the-badge)](https://www.r-project.org/)
[![GitHub Stars](https://img-shields.io/github/stars/USUARIO/REPO?style=social)](https://github.com/USUARIO/REPO/stargazers)

![YALM Generator Portada](www/portada.jpg)

**YALM Generator** es una aplicación web interactiva desarrollada con **R Shiny** diseñada para simplificar y acelerar la creación de cabeceras YAML para documentos de **Quarto (.qmd)** y **R Markdown (.Rmd)**.

---

### 📋 Tabla de Contenidos
1. [Descripción General](#-descripción-general)
2. [Características Principales](#-características-principales)
3. [Instalación y Requisitos](#-instalación-y-requisitos)
4. [Guía de Uso](#-guía-de-uso)
5. [Análisis Técnico del Proyecto](#-análisis-técnico-del-proyecto)
6. [Roadmap Futuro](#-roadmap-futuro)
7. [Cómo Contribuir](#-cómo-contribuir)
8. [Licencia](#-licencia)
9. [Autores y Créditos](#-autores-y-créditos)
10. [Contacto y Soporte](#-contacto-y-soporte)

---

### 🎯 Descripción General

El objetivo principal de esta herramienta es eliminar la necesidad de memorizar la compleja sintaxis de YAML y reducir los errores comunes de formato. Con una interfaz gráfica intuitiva, los usuarios pueden configurar metadatos, formatos de salida y opciones de ejecución, generando en tiempo real una cabecera YAML válida y lista para usar, permitiéndoles centrarse exclusivamente en el contenido de sus documentos.

### ✨ Características Principales

-   **Interfaz Gráfica Intuitiva**: Diseño limpio y organizado en paneles colapsables que facilita la navegación.
-   **Generación en Tiempo Real**: Visualiza la cabecera YAML mientras modificas los parámetros en la interfaz.
-   **Validación Automática**: El sistema verifica la sintaxis del YAML generado y notifica sobre posibles errores.
-   **Soporte Multiformato**: Configura opciones específicas para `HTML`, `PDF`, `Word (docx)` y `Presentaciones (revealjs)`.
-   **Integración con Archivos Existentes**: Carga tus documentos `.qmd`, `.md` o `.Rmd`. La aplicación extraerá el YAML existente para que puedas editarlo y conservará el contenido del cuerpo.
-   **Opciones de Ejecución Globales**: Controla el comportamiento de los bloques de código (`echo`, `eval`, `warning`, `error`) de forma centralizada.
-   **Descarga Directa**: Obtén el documento `.qmd` completo, con la nueva cabecera YAML y el contenido original, listo para renderizar.

### 📦 Instalación y Requisitos

#### Requisitos Previos
-   **R**: Versión 4.0 o superior.
-   **Navegador Web Moderno**: Chrome, Firefox, Safari, Edge.
-   **RStudio (Recomendado)**: Para una mejor experiencia de desarrollo y ejecución.

#### Pasos de Instalación
1.  Clona o descarga este repositorio en tu máquina local.
    ```bash
    git clone [https://github.com/USUARIO/REPO.git](https://github.com/USUARIO/REPO.git)
    ```
2.  Abre R o RStudio y establece el directorio de trabajo en la carpeta del proyecto.
3.  Instala las dependencias necesarias ejecutando el siguiente comando en la consola de R:
    ```r
    install.packages(c("shiny", "shinyjs", "yaml", "bslib"))
    ```
4.  ¡Listo! Ya puedes ejecutar la aplicación.

### 🚀 Guía de Uso

#### Iniciar la Aplicación
Puedes iniciar la aplicación de dos maneras:
1.  **Desde la consola de R** (asegúrate de estar en el directorio raíz del proyecto):
    ```r
    shiny::runApp()
    ```
2.  **Usando RStudio**:
    * Abre el archivo `app.R`.
    * Haz clic en el botón **"Run App"** que aparece en la parte superior del editor.

#### Flujo de Trabajo
1.  **Paso 1: Metadatos**: Completa los campos básicos como título, autor y fecha.
2.  **Paso 2: Formato**: Selecciona el formato de salida (`html`, `pdf`, etc.) y ajusta las opciones específicas.
3.  **Previsualiza el YAML**: En la pestaña **"YAML Preview"**, puedes ver el código generado en tiempo real y copiarlo.
    ![Previsualización de YAML](www/guia.png)
4.  **Paso 3: Carga y Descarga (Opcional)**:
    * Usa el botón **"Upload Document"** para cargar un archivo existente. Los campos se rellenarán automáticamente.
    * Cuando termines, haz clic en **"Download Document"** para guardar un archivo `.qmd` que combina la nueva cabecera y tu contenido.

### 🛠️ Análisis Técnico del Proyecto

#### Estructura de Directorios
