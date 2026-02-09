/**
 * Configuración del tema para Ant Design
 * Paleta profesional financiera — alineada con globals.css
 */

export const colors = {
  primary: {
    main: '#1a3a6b',
    light: '#22518f',
    lighter: '#3d84c6',
    lightest: '#ebf4ff',
  },
  success: {
    main: '#1a5632',
    light: '#2f9e5a',
    lighter: '#68d391',
    bg: '#f0fff4',
  },
  danger: {
    main: '#c53030',
    light: '#fc8181',
    lighter: '#fed7d7',
    bg: '#fff5f5',
  },
  warning: {
    main: '#b7791f',
    light: '#d69e2e',
    lighter: '#fbd38d',
    bg: '#fffaf0',
  },
  neutral: {
    900: '#171f2b',
    800: '#232e3e',
    700: '#374151',
    600: '#5a6678',
    500: '#8492a6',
    400: '#b8c2cc',
    300: '#d3dce6',
    200: '#e8ecf1',
    100: '#f4f6f9',
    50: '#ffffff',
  },
  gradients: {
    primary: 'linear-gradient(135deg, #1a3a6b 0%, #2b6cb0 100%)',
    success: 'linear-gradient(135deg, #1a5632 0%, #2f9e5a 100%)',
    header: 'linear-gradient(135deg, #0c1e3a 0%, #142d54 40%, #1a3a6b 100%)',
    card: 'linear-gradient(180deg, #ffffff 0%, #f9fafb 100%)',
  }
};

// Ant Design light theme tokens
export const lightTheme = {
  token: {
    colorPrimary: colors.primary.light,
    colorSuccess: colors.success.light,
    colorWarning: colors.warning.light,
    colorError: colors.danger.main,
    colorInfo: colors.primary.lighter,

    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontSize: 14,
    fontSizeHeading1: 36,
    fontSizeHeading2: 28,
    fontSizeHeading3: 22,
    fontSizeHeading4: 18,
    fontSizeHeading5: 15,

    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,

    colorBgContainer: colors.neutral[50],
    colorBgLayout: colors.neutral[100],
    colorBgElevated: colors.neutral[50],

    colorText: colors.neutral[900],
    colorTextSecondary: colors.neutral[700],
    colorTextTertiary: colors.neutral[600],
    colorTextDisabled: colors.neutral[500],

    boxShadow: '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
    boxShadowSecondary: '0 4px 8px -2px rgba(0,0,0,0.08), 0 2px 4px -2px rgba(0,0,0,0.04)',

    motionDurationFast: '0.1s',
    motionDurationMid: '0.2s',
    motionDurationSlow: '0.3s',
  },
  components: {
    Layout: {
      headerBg: 'transparent',
      headerHeight: 64,
      headerPadding: '0 32px',
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
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
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
      headerHeight: 64,
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
