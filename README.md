YALM Generator 🚀
YALM Generator es una aplicación web interactiva desarrollada con R Shiny diseñada para simplificar y acelerar la creación de cabeceras YAML para documentos de Quarto (.qmd) y R Markdown (.Rmd).

📋 Tabla de Contenidos
Descripción General

Características Principales

Instalación y Requisitos

Guía de Uso

Análisis Técnico del Proyecto

Roadmap Futuro

Cómo Contribuir

Licencia

Autores y Créditos

Contacto y Soporte

🎯 Descripción General
El objetivo principal de esta herramienta es eliminar la necesidad de memorizar la compleja sintaxis de YAML y reducir los errores comunes de formato. Con una interfaz gráfica intuitiva, los usuarios pueden configurar metadatos, formatos de salida y opciones de ejecución, generando en tiempo real una cabecera YAML válida y lista para usar, permitiéndoles centrarse exclusivamente en el contenido de sus documentos.

✨ Características Principales
Interfaz Gráfica Intuitiva: Diseño limpio y organizado en paneles colapsables que facilita la navegación.

Generación en Tiempo Real: Visualiza la cabecera YAML mientras modificas los parámetros en la interfaz.

Validación Automática: El sistema verifica la sintaxis del YAML generado y notifica sobre posibles errores.

Soporte Multiformato: Configura opciones específicas para HTML, PDF, Word (docx) y Presentaciones (revealjs).

Integración con Archivos Existentes: Carga tus documentos .qmd, .md o .Rmd. La aplicación extraerá el YAML existente para que puedas editarlo y conservará el contenido del cuerpo.

Opciones de Ejecución Globales: Controla el comportamiento de los bloques de código (echo, eval, warning, error) de forma centralizada.

Descarga Directa: Obtén el documento .qmd completo, con la nueva cabecera YAML y el contenido original, listo para renderizar.

📦 Instalación y Requisitos
Requisitos Previos
R: Versión 4.0 o superior.

Navegador Web Moderno: Chrome, Firefox, Safari, Edge.

RStudio (Recomendado): Para una mejor experiencia de desarrollo y ejecución.

Pasos de Instalación
Clona o descarga este repositorio en tu máquina local.

git clone [https://github.com/USUARIO/REPO.git](https://github.com/USUARIO/REPO.git)

Abre R o RStudio y establece el directorio de trabajo en la carpeta del proyecto.

Instala las dependencias necesarias ejecutando el siguiente comando en la consola de R:

install.packages(c("shiny", "shinyjs", "yaml", "bslib"))

¡Listo! Ya puedes ejecutar la aplicación.

🚀 Guía de Uso
Iniciar la Aplicación
Puedes iniciar la aplicación de dos maneras:

Desde la consola de R (asegúrate de estar en el directorio raíz del proyecto):

shiny::runApp()

Usando RStudio:

Abre el archivo app.R.

Haz clic en el botón "Run App" que aparece en la parte superior del editor.

Flujo de Trabajo
Paso 1: Metadatos: Completa los campos básicos como título, autor y fecha.

Paso 2: Formato: Selecciona el formato de salida (html, pdf, etc.) y ajusta las opciones específicas.

Previsualiza el YAML: En la pestaña "YAML Preview", puedes ver el código generado en tiempo real y copiarlo.

Paso 3: Carga y Descarga (Opcional):

Usa el botón "Upload Document" para cargar un archivo existente. Los campos se rellenarán automáticamente.

Cuando termines, haz clic en "Download Document" para guardar un archivo .qmd que combina la nueva cabecera y tu contenido.

🛠️ Análisis Técnico del Proyecto
Estructura de Directorios
YALM_Generator/
├── app.R                 # Aplicación principal de Shiny
├── modules/              # Módulos de Shiny
│   ├── doc_actions_module.R
│   └── fileUploadUI.R
└── www/                  # Recursos estáticos (assets)
    ├── styles.css
    ├── portada.jpg
    └── ...

Arquitectura y Componentes
Aplicación Principal (app.R): Contiene el framework de la aplicación Shiny, utilizando bslib para una UI moderna. Implementa la lógica central para la generación y validación de YAML.

Estructura Modular (/modules):

fileUploadUI.R: Gestiona la carga de documentos.

doc_actions_module.R: Maneja las acciones de usuario como "descargar" o "limpiar formulario".

Recursos Web (/www): Define un tema oscuro profesional con styles.css para mejorar la legibilidad.

🗺️ Roadmap Futuro
[ ] Guardar Plantillas: Permitir a los usuarios guardar y cargar configuraciones de YAML frecuentes.

[ ] Soporte para más Formatos: Añadir configuraciones para beamer, bookdown, etc.

[ ] Mejoras en la UI/UX: Incorporar un editor de texto para el cuerpo del documento directamente en la app.

[ ] Internacionalización: Soporte para múltiples idiomas en la interfaz.

🤝 Cómo Contribuir
¡Las contribuciones son bienvenidas! Si deseas mejorar YALM Generator, por favor sigue estos pasos:

Haz un Fork del repositorio.

Crea una nueva Rama: git checkout -b feature/nueva-funcionalidad.

Realiza tus Cambios y asegúrate de que el código esté bien documentado.

Envía un Pull Request con una descripción clara de tus aportes.

Para reportar errores o sugerir nuevas características, por favor crea un Issue en el repositorio.

📄 Licencia
Este proyecto está distribuido bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

👨‍💻 Autores y Créditos
Este proyecto fue desarrollado como una herramienta para la comunidad de R y Quarto.

Tecnologías Clave:

Shiny: Framework para aplicaciones web en R.

yaml: Para el parseo y generación de YAML.

bslib: Para la creación de temas modernos.

shinyjs: Para ejecutar código JavaScript personalizado.

🆘 Contacto y Soporte
Si tienes preguntas, encuentras un error o necesitas ayuda, por favor abre un Issue en el repositorio oficial de GitHub.
