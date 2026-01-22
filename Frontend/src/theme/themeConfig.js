/**
 * Configuración del tema personalizado para Ant Design
 * Paleta de colores profesional para aplicación financiera
 */

// Colores principales de la aplicación
export const colors = {
  // Colores primarios - Azul profesional financiero
  primary: {
    main: '#1a365d',      // Azul oscuro profesional
    light: '#2c5282',     // Azul medio
    lighter: '#4299e1',   // Azul claro
    lightest: '#ebf8ff',  // Azul muy claro (backgrounds)
  },
  
  // Colores de acento - Verde para ganancias, éxito
  success: {
    main: '#22543d',
    light: '#38a169',
    lighter: '#68d391',
    bg: '#f0fff4',
  },
  
  // Colores de advertencia/pérdida - Rojo
  danger: {
    main: '#c53030',
    light: '#fc8181',
    lighter: '#fed7d7',
    bg: '#fff5f5',
  },
  
  // Colores de advertencia - Naranja/Amarillo
  warning: {
    main: '#c05621',
    light: '#ed8936',
    lighter: '#fbd38d',
    bg: '#fffaf0',
  },
  
  // Neutrales
  neutral: {
    900: '#1a202c',   // Textos principales
    800: '#2d3748',   // Textos secundarios
    700: '#4a5568',   // Textos terciarios
    600: '#718096',   // Textos deshabilitados
    500: '#a0aec0',   // Bordes oscuros
    400: '#cbd5e0',   // Bordes claros
    300: '#e2e8f0',   // Divisores
    200: '#edf2f7',   // Fondos hover
    100: '#f7fafc',   // Fondos principales
    50: '#ffffff',    // Blanco puro
  },
  
  // Gradientes
  gradients: {
    primary: 'linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #4299e1 100%)',
    success: 'linear-gradient(135deg, #22543d 0%, #38a169 100%)',
    header: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1a365d 100%)',
    card: 'linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)',
  }
};

// Configuración del tema de Ant Design
export const lightTheme = {
  token: {
    // Colores base
    colorPrimary: colors.primary.light,
    colorSuccess: colors.success.light,
    colorWarning: colors.warning.light,
    colorError: colors.danger.main,
    colorInfo: colors.primary.lighter,
    
    // Tipografía
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    fontSize: 14,
    fontSizeHeading1: 38,
    fontSizeHeading2: 30,
    fontSizeHeading3: 24,
    fontSizeHeading4: 20,
    fontSizeHeading5: 16,
    
    // Bordes
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,
    
    // Colores de fondo
    colorBgContainer: colors.neutral[50],
    colorBgLayout: colors.neutral[100],
    colorBgElevated: colors.neutral[50],
    
    // Colores de texto
    colorText: colors.neutral[900],
    colorTextSecondary: colors.neutral[700],
    colorTextTertiary: colors.neutral[600],
    colorTextDisabled: colors.neutral[500],
    
    // Sombras
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    boxShadowSecondary: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    
    // Motion
    motionDurationFast: '0.1s',
    motionDurationMid: '0.2s',
    motionDurationSlow: '0.3s',
  },
  components: {
    Layout: {
      headerBg: 'transparent', // El header usará gradiente CSS
      headerHeight: 72,
      headerPadding: '0 24px',
      bodyBg: colors.neutral[100],
      footerBg: colors.neutral[50],
    },
    Card: {
      headerBg: 'transparent',
      paddingLG: 24,
      borderRadiusLG: 12,
    },
    Button: {
      primaryShadow: '0 2px 4px rgba(26, 54, 93, 0.2)',
      defaultBorderColor: colors.neutral[400],
    },
    Input: {
      activeBorderColor: colors.primary.light,
      hoverBorderColor: colors.primary.lighter,
    },
    Select: {
      optionSelectedBg: colors.primary.lightest,
    },
    Tabs: {
      inkBarColor: colors.primary.light,
      itemActiveColor: colors.primary.main,
      itemHoverColor: colors.primary.light,
    },
    Progress: {
      defaultColor: colors.primary.light,
    },
    Tag: {
      defaultBg: colors.neutral[200],
    },
    Alert: {
      successBorderColor: colors.success.lighter,
      successIcon: colors.success.light,
      warningBorderColor: colors.warning.lighter,
      warningIcon: colors.warning.light,
      errorBorderColor: colors.danger.lighter,
      errorIcon: colors.danger.main,
      infoBorderColor: colors.primary.lighter,
      infoIcon: colors.primary.light,
    },
  }
};

export const darkTheme = {
  token: {
    // Colores base para modo oscuro
    colorPrimary: '#4299e1',
    colorSuccess: '#68d391',
    colorWarning: '#ed8936',
    colorError: '#fc8181',
    colorInfo: '#63b3ed',
    
    // Tipografía
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    fontSize: 14,
    
    // Bordes
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,
    
    // Colores de fondo oscuros
    colorBgContainer: '#1a202c',
    colorBgLayout: '#0f1419',
    colorBgElevated: '#2d3748',
    
    // Colores de texto para modo oscuro
    colorText: '#f7fafc',
    colorTextSecondary: '#e2e8f0',
    colorTextTertiary: '#a0aec0',
    colorTextDisabled: '#718096',
    
    // Bordes para modo oscuro
    colorBorder: '#4a5568',
    colorBorderSecondary: '#2d3748',
  },
  components: {
    Layout: {
      headerBg: 'transparent',
      headerHeight: 72,
      bodyBg: '#0f1419',
      footerBg: '#1a202c',
    },
    Card: {
      colorBgContainer: '#1a202c',
      headerBg: 'transparent',
    },
    Modal: {
      contentBg: '#1a202c',
      headerBg: '#1a202c',
    },
  }
};

// Exportar tema por defecto
export default lightTheme;
