/**
 * Context para gestión de estado global de la aplicación
 * Centraliza la configuración, entrenamiento y pronósticos
 */
import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { message } from 'antd';
import { getAvailableModels } from '../services/api';

// Crear el contexto
const AppContext = createContext(null);

// Constantes
const MIN_CALENDAR_DAYS = 760;
const POLLING_TIMEOUT_MS = 600000; // 10 minutos

// Lista de tickers disponibles
const AVAILABLE_TICKERS = [
  // Tech Giants
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD',
  // Financial
  'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'V', 'MA', 'AXP',
  // Healthcare & Pharma
  'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
  // Energy & Oil
  'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
  // Consumer & Retail
  'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'DIS', 'COST',
  // Industrial & Aerospace
  'BA', 'CAT', 'GE', 'LMT', 'RTX', 'HON', 'UPS', 'FDX', 'MMM',
  // Telecommunications
  'T', 'VZ', 'TMUS', 'CMCSA',
  // Latin America & Emerging Markets
  'NU', 'VALE', 'ITUB', 'BBD', 'PBR', 'GGAL', 'YPF', 'MELI', 'GLOB',
  // ETFs & Indices
  'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VEA', 'VWO', 'SPE',
  // Commodities & Currencies
  'GLD', 'SLV', 'USO', 'UNG', 'DXY', 'UUP',
  // Volatility & Alternative
  'VIX', 'UVXY', 'SQQQ', 'TQQQ',
  // Cryptocurrencies
  'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD', 'MATIC-USD',
  // Real Estate
  'VNQ', 'REZ', 'IYR',
  // International Banks
  'CIB', 'BABA', 'TSM', 'ASML'
];

const AVAILABLE_MODEL_TYPES = ['rf', 'lstm', 'xgboost', 'prophet'];

// Configuración inicial
const getInitialConfig = () => ({
  selectedModelType: 'rf',
  selectedTicker: 'NU',
  startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 3)),
  endDate: new Date(),
  forecastHorizon: 10,
  nLags: 10,
  targetCol: 'Close',
  training_period_preset: '3_years',
  custom_start_date: null,
  custom_end_date: null,
});

// Provider del contexto
export function AppProvider({ children }) {
  // === ESTADO DE CONFIGURACIÓN ===
  const [config, setConfig] = useState(getInitialConfig());
  const [currentTheme, setCurrentTheme] = useState('light');
  
  // === ESTADO DE DATOS ===
  const [historicalData, setHistoricalData] = useState({ dates: [], values: [] });
  const [forecastData, setForecastData] = useState([]);
  const [trainingResults, setTrainingResults] = useState({});
  const [lastModelUsed, setLastModelUsed] = useState('');
  const [residualsData, setResidualsData] = useState({
    dates: [],
    values: [],
    acf: null,
    pacf: null
  });
  
  // === ESTADO DE ENTRENAMIENTOS ACTIVOS ===
  const [activeTrainingJobs, setActiveTrainingJobs] = useState({});
  const [pollingIntervals, setPollingIntervals] = useState({});
  
  // === ESTADO DE MODELOS DISPONIBLES ===
  const [availableModels, setAvailableModels] = useState({});
  const [availableModelsLoading, setAvailableModelsLoading] = useState(false);
  const [availableModelsError, setAvailableModelsError] = useState(null);
  
  // === ESTADO DE CARGA DE MODELOS ===
  const [loadedModelData, setLoadedModelData] = useState(null);
  const [isLoadingModelData, setIsLoadingModelData] = useState(false);
  const [loadedModelError, setLoadedModelError] = useState(null);
  
  // === ESTADO DE VALIDACIÓN ===
  const [dateRangeWarning, setDateRangeWarning] = useState('');
  
  // === ESTADO DE UI ===
  const [helpModalVisible, setHelpModalVisible] = useState(false);

  // Referencias para cleanup
  const pollingIntervalsRef = useRef(pollingIntervals);
  pollingIntervalsRef.current = pollingIntervals;

  // === HELPERS ===
  const getTrainingKey = useCallback((ticker, modelType) => {
    return `${ticker}-${modelType}`;
  }, []);

  // === GESTIÓN DE TRABAJOS DE ENTRENAMIENTO ===
  const addActiveTrainingJob = useCallback((ticker, modelType, jobData) => {
    const key = getTrainingKey(ticker, modelType);
    setActiveTrainingJobs(prev => ({
      ...prev,
      [key]: {
        key,
        ticker,
        modelType,
        startTime: new Date(),
        ...jobData
      }
    }));
  }, [getTrainingKey]);

  const updateActiveTrainingJob = useCallback((ticker, modelType, updates) => {
    const key = getTrainingKey(ticker, modelType);
    setActiveTrainingJobs(prev => {
      if (!prev[key]) return prev;
      return {
        ...prev,
        [key]: { ...prev[key], ...updates }
      };
    });
  }, [getTrainingKey]);

  const removeActiveTrainingJob = useCallback((ticker, modelType) => {
    const key = getTrainingKey(ticker, modelType);
    setActiveTrainingJobs(prev => {
      const newJobs = { ...prev };
      delete newJobs[key];
      return newJobs;
    });
    
    // Limpiar polling
    if (pollingIntervalsRef.current[key]) {
      clearInterval(pollingIntervalsRef.current[key]);
      setPollingIntervals(prev => {
        const newIntervals = { ...prev };
        delete newIntervals[key];
        return newIntervals;
      });
    }
  }, [getTrainingKey]);

  const isCurrentConfigurationTraining = useCallback(() => {
    const key = getTrainingKey(config.selectedTicker, config.selectedModelType);
    const job = activeTrainingJobs[key];
    return job && ['queued', 'running', 'submitting'].includes(job.status);
  }, [config.selectedTicker, config.selectedModelType, activeTrainingJobs, getTrainingKey]);

  const getActiveTrainingJobsCount = useCallback(() => {
    return Object.keys(activeTrainingJobs).length;
  }, [activeTrainingJobs]);

  const getActiveTrainingJobsForDisplay = useCallback(() => {
    return Object.values(activeTrainingJobs).map(job => ({
      ...job,
      key: job.key || getTrainingKey(job.ticker, job.modelType)
    }));
  }, [activeTrainingJobs, getTrainingKey]);

  // === GESTIÓN DE CONFIGURACIÓN ===
  const handleConfigChange = useCallback((newConfig) => {
    setConfig(prev => {
      const updatedConfig = { ...prev, ...newConfig };
      
      // Convertir fechas a objetos Date si es necesario
      if (newConfig.startDate && !(newConfig.startDate instanceof Date)) {
        updatedConfig.startDate = new Date(newConfig.startDate);
      }
      if (newConfig.endDate && !(newConfig.endDate instanceof Date)) {
        updatedConfig.endDate = new Date(newConfig.endDate);
      }
      if (newConfig.custom_start_date !== undefined) {
        updatedConfig.custom_start_date = newConfig.custom_start_date 
          ? new Date(newConfig.custom_start_date) 
          : null;
      }
      if (newConfig.custom_end_date !== undefined) {
        updatedConfig.custom_end_date = newConfig.custom_end_date 
          ? new Date(newConfig.custom_end_date) 
          : null;
      }
      
      return updatedConfig;
    });
  }, []);

  // === GESTIÓN DE TEMA ===
  const toggleTheme = useCallback((checked) => {
    setCurrentTheme(checked ? 'dark' : 'light');
  }, []);

  // === EFECTOS ===
  
  // Validar rango de fechas
  useEffect(() => {
    const { startDate, endDate } = config;
    if (startDate instanceof Date && endDate instanceof Date && !isNaN(startDate) && !isNaN(endDate)) {
      if (endDate <= startDate) {
        setDateRangeWarning('Advertencia: La fecha de fin debe ser posterior a la fecha de inicio.');
        return;
      }
      const diffTime = Math.abs(endDate - startDate);
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      if (diffDays < MIN_CALENDAR_DAYS) {
        setDateRangeWarning(`Advertencia: El rango seleccionado (${diffDays} días) podría ser muy corto. Se recomiendan al menos ${MIN_CALENDAR_DAYS} días (~25 meses).`);
      } else {
        setDateRangeWarning('');
      }
    } else {
      setDateRangeWarning('Advertencia: Las fechas seleccionadas no son válidas.');
    }
  }, [config.startDate, config.endDate]);

  // Cargar modelos disponibles cuando cambia el tipo de modelo
  useEffect(() => {
    if (config.selectedModelType) {
      setAvailableModelsLoading(true);
      setAvailableModelsError(null);
      
      getAvailableModels(config.selectedModelType)
        .then(response => {
          setAvailableModels(prev => ({
            ...prev,
            [config.selectedModelType]: response
          }));
        })
        .catch(error => {
          console.error(`Error loading available models:`, error);
          setAvailableModelsError(error.message);
          setAvailableModels(prev => ({
            ...prev,
            [config.selectedModelType]: null
          }));
        })
        .finally(() => {
          setAvailableModelsLoading(false);
        });
    }
  }, [config.selectedModelType]);

  // Limpiar datos cuando cambia el ticker
  useEffect(() => {
    setHistoricalData({ dates: [], values: [] });
    setForecastData([]);
    setLastModelUsed('');
    setLoadedModelData(null);
    setLoadedModelError(null);
  }, [config.selectedTicker]);

  // Cleanup polling al desmontar
  useEffect(() => {
    return () => {
      Object.values(pollingIntervalsRef.current).forEach(intervalId => {
        clearInterval(intervalId);
      });
    };
  }, []);

  // === VALOR DEL CONTEXTO ===
  const contextValue = {
    // Configuración
    config,
    setConfig,
    handleConfigChange,
    
    // Tema
    currentTheme,
    toggleTheme,
    
    // Datos
    historicalData,
    setHistoricalData,
    forecastData,
    setForecastData,
    trainingResults,
    setTrainingResults,
    lastModelUsed,
    setLastModelUsed,
    residualsData,
    setResidualsData,
    
    // Trabajos de entrenamiento
    activeTrainingJobs,
    addActiveTrainingJob,
    updateActiveTrainingJob,
    removeActiveTrainingJob,
    isCurrentConfigurationTraining,
    getActiveTrainingJobsCount,
    getActiveTrainingJobsForDisplay,
    pollingIntervals,
    setPollingIntervals,
    
    // Modelos disponibles
    availableModels,
    setAvailableModels,
    availableModelsLoading,
    availableModelsError,
    
    // Carga de modelos
    loadedModelData,
    setLoadedModelData,
    isLoadingModelData,
    setIsLoadingModelData,
    loadedModelError,
    setLoadedModelError,
    
    // Validación
    dateRangeWarning,
    
    // UI
    helpModalVisible,
    setHelpModalVisible,
    
    // Constantes
    availableTickers: AVAILABLE_TICKERS,
    availableModelTypes: AVAILABLE_MODEL_TYPES,
    minCalendarDays: MIN_CALENDAR_DAYS,
    pollingTimeoutMs: POLLING_TIMEOUT_MS,
    
    // Helpers
    getTrainingKey,
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
}

// Hook personalizado para usar el contexto
export function useAppContext() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext debe usarse dentro de un AppProvider');
  }
  return context;
}

export default AppContext;
