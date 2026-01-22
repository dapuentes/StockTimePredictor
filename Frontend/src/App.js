import React, { useState, useCallback, useEffect } from 'react';
import dayjs from 'dayjs';
import { Layout, Row, Col, Card, Spin, Alert, Tabs, message, Button, Switch, ConfigProvider, theme as antdTheme, Modal  } from 'antd';
import { SunOutlined, MoonOutlined } from '@ant-design/icons';
import Papa from 'papaparse';
import ResidualsDisplay from './components/ResidualsDisplay';
import ConfigurationPanel from './components/ConfigurationPanel_AntD';
import GraphDisplay from './components/GraphDisplay';
import MetricsDisplay from './components/MetricsDisplay_AntD';
import ModelComparisonTable from './components/ModelComparisonTable';
import ModelDetailsDisplay from './components/ModelDetailsDisplay';
import HelpModal from './components/HelpModal';
import ActiveTrainingJobs from './components/ActiveTrainingJobs';
import ShapExplainer from './components/ShapExplainer';
import EnsemblePredictor from './components/EnsemblePredictor';
import { useTrainModelMutation, useGenerateForecastMutation, useTrainingStatusMutation } from './hooks/useApiMutations'; // Ajusta el path
import { getAvailableModels } from './services/api';
import { parseMetadata } from './utils/pythonUtils'; // Import the metadata parsing utility
import './App.css';

// Extraer componentes de Layout para claridad
const { Header, Content, Footer } = Layout;

const min_calendar_days = 760; // Mínimo de días calendario para el rango de fechas

function App() {
    const [helpModalVisible, setHelpModalVisible] = useState(false); // Estado para el modal de ayuda

    const [config, setConfig] = useState({
        selectedModelType: 'rf',
        selectedTicker: 'NU',
        startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 3)), // 3 años atrás
        endDate: new Date(),
        forecastHorizon: 10,
        nLags: 10,
        targetCol: 'Close',
        training_period_preset: '3_years', // Preset por defecto: "Últimos 3 años"
        custom_start_date: null, // Fecha de inicio personalizada
        custom_end_date: null // Fecha de fin personalizada
    });
    const [historicalData, setHistoricalData] = useState({ dates: [], values: [] });
    const [forecastData, setForecastData] = useState([]);
    const [trainingResults, setTrainingResults] = useState({});
    const [lastModelUsed, setLastModelUsed] = useState('');
    
    // Enhanced state for multiple parallel trainings
    const [activeTrainingJobs, setActiveTrainingJobs] = useState({}); // { "ticker-modelType": { jobId, status, message, startTime, config } }
    const [currentForecastJob, setCurrentForecastJob] = useState(null); // Single forecast at a time
    const [pollingIntervals, setPollingIntervals] = useState({}); // { "ticker-modelType": intervalId }
    
    // Legacy states for backward compatibility (will be removed gradually)
    const [currentTrainingJob, setCurrentTrainingJob] = useState(null);
    const [trainingStatus, setTrainingStatus] = useState('idle');
    const [pollingIntervalId, setPollingIntervalId] = useState(null);
    const [currentJobId, setCurrentJobId] = useState(null);
    const [isPollingStatus, setIsPollingStatus] = useState(false);
    const [trainingStatusMessage, setTrainingStatusMessage] = useState('');
    const [pollingError, setPollingError] = useState(null);
    const [forecastToTrainFlow, setForecastToTrainFlow] = useState(null);
    
    // State for loaded model data (when clicking on trained models)
    const [loadedModelData, setLoadedModelData] = useState(null); // Contains metadata and training results
    const [isLoadingModelData, setIsLoadingModelData] = useState(false);
    const [loadedModelError, setLoadedModelError] = useState(null);

    const [availableModelTypes] = useState(['rf', 'lstm', 'xgboost', 'prophet']);
    const [availableModels, setAvailableModels] = useState({});
    const [availableModelsLoading, setAvailableModelsLoading] = useState(false);
    const [availableModelsError, setAvailableModelsError] = useState(null);
    const [availableTickers] = useState([
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
    ]);    const [dateRangeWarning, setDateRangeWarning] = useState('');
    const [currentTheme, setCurrentTheme] = useState('light');
    const toggleTheme = (checked) => {
        setCurrentTheme(checked ? 'dark' : 'light');
    };

    // --- Helper functions for managing multiple training jobs ---
    const getTrainingKey = useCallback((ticker, modelType) => {
        return `${ticker}-${modelType}`;
    }, []);

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
                [key]: {
                    ...prev[key],
                    ...updates
                }
            };
        });
    }, [getTrainingKey]);    const removeActiveTrainingJob = useCallback((ticker, modelType) => {
        const key = getTrainingKey(ticker, modelType);
        setActiveTrainingJobs(prev => {
            const newJobs = { ...prev };
            delete newJobs[key];
            return newJobs;
        });
        
        // Also stop polling for this job - will be defined later
        if (pollingIntervals[key]) {
            clearInterval(pollingIntervals[key]);
            setPollingIntervals(prev => {
                const newIntervals = { ...prev };
                delete newIntervals[key];
                return newIntervals;
            });
        }
    }, [getTrainingKey, pollingIntervals]);

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

    // --- Usar los hooks de mutación ---
    const trainMutation = useTrainModelMutation({
        onSuccessCallback: (jobData, variables) => { // jobData now contains { job_id, status, message }
            const { job_id, status, message: jobMessage } = jobData;
            const modelType = variables.modelType;
            const config = variables.config;
            const ticker = config.selectedTicker;
            
            // Add to active training jobs using new system
            addActiveTrainingJob(ticker, modelType, {
                jobId: job_id,
                status: 'queued',
                message: `Entrenamiento iniciado (ID: ${job_id.slice(0, 8)}...)`,
                config: config
            });
            
            // Legacy support - only update if this is the current configuration
            if (ticker === config.selectedTicker && modelType === config.selectedModelType) {
                setCurrentTrainingJob({
                    jobId: job_id,
                    modelType: modelType,
                    config: config,
                    startTime: new Date()
                });
                setCurrentJobId(job_id);
                setIsPollingStatus(true);
                setTrainingStatus('queued');
                setTrainingStatusMessage(`Entrenamiento iniciado con ID: ${job_id.slice(0, 8)}... Verificando estado...`);
            }
            
            // Start polling for status with 5-second interval
            startStatusPollingForJob(ticker, modelType, job_id, 5);
        },        
        onErrorCallback: (err, variables) => {
            const modelType = variables?.modelType;
            const ticker = variables?.config?.selectedTicker;
            
            if (ticker && modelType) {
                removeActiveTrainingJob(ticker, modelType);
            }
            
            // Legacy support - only update if this is the current configuration
            if (ticker === config.selectedTicker && modelType === config.selectedModelType) {
                setHistoricalData({ dates: [], values: [] }); // Limpiar datos en error
                setForecastData([]);            
                setTrainingResults({});
                setLastModelUsed('');
                setTrainingStatus('failed');
                setCurrentTrainingJob(null);
                setCurrentJobId(null);
                setIsPollingStatus(false);
                setTrainingStatusMessage('Error al iniciar el entrenamiento');
                setPollingError(null);
            }
            
            message.error(`Error al iniciar entrenamiento para ${ticker} (${modelType.toUpperCase()}): ${err.message}`);
        }
    });const forecastMutation = useGenerateForecastMutation({
        onSuccessCallback: (forecastResult) => {
            setHistoricalData({
                dates: forecastResult.historical_dates || [],
                values: forecastResult.historical_values || []
            });
            setForecastData(forecastResult.predictions || []);
            setLastModelUsed(forecastResult.model_used || `Predicted with ${config.selectedModelType.toUpperCase()}`);
        },
        onErrorCallback: (err, variables) => {
            setHistoricalData({ dates: [], values: [] }); // Limpiar datos en error
            setForecastData([]);
            
            // Check if it's a 404 error indicating model not found
            const isModelNotFoundError = err?.response?.status === 404 || 
                                       (err?.message && err.message.includes('modelo no encontrado')) ||
                                       (err?.response?.data && typeof err.response.data === 'string' && err.response.data.includes('no encontrado')) ||
                                       (err?.response?.data?.detail && err.response.data.detail.includes('no encontrado'));
            
            if (isModelNotFoundError && variables) {
                const { modelType, config: forecastConfig } = variables;
                const ticker = forecastConfig.selectedTicker;
                const modelTypeUpper = modelType.toUpperCase();
                
                // Store the forecast request for potential training
                setForecastToTrainFlow({ modelType, config: forecastConfig });
                
                // Show modal asking user if they want to train
                Modal.confirm({
                    title: 'Modelo no encontrado',
                    content: `No se encontró un modelo entrenado de tipo ${modelTypeUpper} para el ticker ${ticker}. ¿Deseas entrenar un modelo ahora?`,
                    okText: 'Entrenar ahora',
                    cancelText: 'Cancelar',
                    onOk: () => {
                        message.info(`Iniciando entrenamiento de modelo ${modelTypeUpper} para ${ticker}...`);
                        handleTrain(modelType, forecastConfig);
                        setForecastToTrainFlow(null); // Clear the flow state
                    },
                    onCancel: () => {
                        message.info('Entrenamiento cancelado por el usuario.');
                        setForecastToTrainFlow(null); // Clear the flow state
                    }
                });
            }
        }
    });    // Hook for checking training status
    const statusMutation = useTrainingStatusMutation({
        onSuccessCallback: (statusData) => {
            handleTrainingStatusUpdate(statusData);
        },        
        onErrorCallback: (error) => {
            // Network error or API failure - stop all polling
            console.error('Error checking training status:', error);
            
            // Stop all polling intervals
            Object.keys(pollingIntervals).forEach(key => {
                clearInterval(pollingIntervals[key]);
            });
            setPollingIntervals({});
            
            // Legacy support
            clearInterval(pollingIntervalId);
            setPollingIntervalId(null);
            setIsPollingStatus(false);
            setTrainingStatus('failed');
            setCurrentTrainingJob(null);
            
            // Store the error details
            const errorMessage = error?.message || error?.response?.data?.message || 'Error de red o comunicación con la API';
            setPollingError(errorMessage);
            setTrainingStatusMessage('');
            
            // Show error message to user
            message.error(`Error consultando el estado del entrenamiento: ${errorMessage}`);
        }
    });// Enhanced polling function for individual jobs
    const startStatusPollingForJob = useCallback((ticker, modelType, jobId, intervalSeconds = 5) => {
        const key = getTrainingKey(ticker, modelType);
        
        // Validate interval parameter (5-10 seconds range)
        const validInterval = Math.max(5, Math.min(10, intervalSeconds));
        const intervalMs = validInterval * 1000;

        // Clear any existing polling interval for this job
        if (pollingIntervals[key]) {
            clearInterval(pollingIntervals[key]);
        }

        // Start new polling interval
        const intervalId = setInterval(() => {
            statusMutation.mutate({ modelType, jobId });
        }, intervalMs);

        setPollingIntervals(prev => ({
            ...prev,
            [key]: intervalId
        }));
        
        // Set a timeout to stop polling after 10 minutes
        setTimeout(() => {
            if (pollingIntervals[key] === intervalId) {
                stopPollingForJob(ticker, modelType);
                updateActiveTrainingJob(ticker, modelType, {
                    status: 'timeout',
                    message: 'Tiempo de espera agotado'
                });
                message.warning(`El tiempo de espera del entrenamiento para ${ticker} (${modelType.toUpperCase()}) ha expirado.`);
            }
        }, 600000); // 10 minutes

        console.log(`Started status polling for ${ticker}-${modelType} job ${jobId} every ${validInterval} seconds`);    
    }, [statusMutation, pollingIntervals]);

    // Function to stop polling for a specific job
    const stopPollingForJob = useCallback((ticker, modelType) => {
        const key = getTrainingKey(ticker, modelType);
        if (pollingIntervals[key]) {
            clearInterval(pollingIntervals[key]);
            setPollingIntervals(prev => {
                const newIntervals = { ...prev };
                delete newIntervals[key];
                return newIntervals;
            });
        }
    }, [pollingIntervals]);

    // Configurable function to start polling for training status (legacy support)
    const startStatusPolling = useCallback((modelType, jobId, intervalSeconds = 5) => {
        // This is for backward compatibility with current configuration
        return startStatusPollingForJob(config.selectedTicker, modelType, jobId, intervalSeconds);
    }, [startStatusPollingForJob, config.selectedTicker]);

    // Function to stop polling
    const stopTrainingStatusPolling = useCallback(() => {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            setPollingIntervalId(null);
        }
        setIsPollingStatus(false);
        setCurrentJobId(null);
        setPollingError(null);
        // Clear status message when polling stops
        setTimeout(() => setTrainingStatusMessage(''), 2000); // Clear after 2 seconds
    }, [pollingIntervalId]);    // Function to handle training status updates - Enhanced for multi-job system
    const handleTrainingStatusUpdate = useCallback((statusData) => {
        const { 
            status, 
            result, 
            error, 
            progress, 
            current_step, 
            training_step, 
            total_training_steps, 
            sub_progress,
            message: responseMessage,
            job_id
        } = statusData;

        // Find which job this status update is for
        let targetJob = null;
        let targetTicker = null;
        let targetModelType = null;

        // Try to find the job by job_id in active jobs
        if (job_id) {
            Object.values(activeTrainingJobs).forEach(job => {
                if (job.jobId === job_id) {
                    targetJob = job;
                    targetTicker = job.ticker;
                    targetModelType = job.modelType;
                }
            });
        }

        // Fallback to current training job for legacy support
        if (!targetJob && currentTrainingJob) {
            targetJob = currentTrainingJob;
            targetTicker = currentTrainingJob.config?.selectedTicker;
            targetModelType = currentTrainingJob.modelType;
        }

        if (!targetTicker || !targetModelType) {
            console.warn('Could not identify job for status update:', statusData);
            return;
        }

        if (status === 'SUCCESS' || status === 'completed') {
            // Training completed successfully
            if (targetJob && result) {
                const startDate = targetJob.config.startDate.toISOString().split('T')[0];
                const endDate = targetJob.config.endDate.toISOString().split('T')[0];
                const runId = `${targetModelType}-${targetTicker}-${startDate}-${endDate}`;

                // Update the job status to completed
                updateActiveTrainingJob(targetTicker, targetModelType, {
                    status: 'completed',
                    message: 'Entrenamiento completado exitosamente',
                    result: result
                });

                // Update global data (for the currently selected configuration if it matches)
                if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
                    // Set residuals data
                    setResidualsData({
                        dates: result.residual_dates || [],
                        values: result.residuals || [],
                        acf: result.acf || null,
                        pacf: result.pacf || null
                    });

                    // Update historical data if included in training result
                    if (result.historical_dates && result.historical_values) {
                        setHistoricalData({
                            dates: result.historical_dates,
                            values: result.historical_values
                        });
                    }

                    setLastModelUsed(result.model_path || `Trained ${targetModelType.toUpperCase()}`);
                    
                    // Legacy support
                    setTrainingStatus('completed');
                    setCurrentTrainingJob(null);
                    setTrainingStatusMessage('');
                }

                // Set training results
                setTrainingResults(prevResults => ({
                    ...prevResults,
                    [runId]: {
                        id: runId,
                        modelType: targetModelType.toUpperCase(),
                        ticker: targetTicker,
                        dateRange: `${startDate} / ${endDate}`,
                        metrics: result.metrics || {},
                        modelPath: result.model_path || 'N/A',
                        timestamp: new Date().toISOString(),
                        bestParams: result.best_params || {},
                        featureNames: result.features_names || []
                    }
                }));

                // Check if we should automatically generate forecast after training
                if (forecastToTrainFlow && forecastToTrainFlow.modelType === targetModelType) {
                    const { config: forecastConfig } = forecastToTrainFlow;
                    
                    // Show message and automatically trigger forecast
                    message.success(`¡Modelo ${targetModelType.toUpperCase()} entrenado exitosamente! Generando pronóstico automáticamente...`);
                    
                    // Clear the forecast-to-train flow state
                    setForecastToTrainFlow(null);
                    
                    // Trigger forecast with a small delay to allow UI to update
                    setTimeout(() => {
                        forecastMutation.mutate({ modelType: targetModelType, config: forecastConfig });
                    }, 1000);
                } else {
                    // Show normal success message
                    message.success(`¡Modelo ${targetModelType.toUpperCase()} para ${targetTicker} entrenado exitosamente!`);
                }

                // Remove job after a delay to show completion status
                setTimeout(() => {
                    removeActiveTrainingJob(targetTicker, targetModelType);
                }, 3000);
            }
        } else if (status === 'FAILURE' || status === 'failed') {
            // Training failed
            const errorMessage = error || responseMessage || result?.error || 'Error desconocido';
            
            updateActiveTrainingJob(targetTicker, targetModelType, {
                status: 'failed',
                message: `Error: ${errorMessage}`,
                error: errorMessage
            });

            // Legacy support for current configuration
            if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
                setTrainingStatus('failed');
                setCurrentTrainingJob(null);
                setPollingError(errorMessage);
                setTrainingStatusMessage('');
            }
            
            // Clear forecast-to-train flow if training fails
            if (forecastToTrainFlow && forecastToTrainFlow.modelType === targetModelType) {
                setForecastToTrainFlow(null);
                message.error(`Entrenamiento falló: ${errorMessage}. El pronóstico automático se ha cancelado.`);
            } else {
                message.error(`Entrenamiento de ${targetModelType.toUpperCase()} para ${targetTicker} falló: ${errorMessage}`);
            }

        } else if (status === 'PROGRESS' || status === 'running' || status === 'PENDING') {
            // Training in progress
            let statusMessage = '';
            if (progress) {
                statusMessage = `Progreso: ${progress}%`;
            } else if (responseMessage) {
                statusMessage = responseMessage;
            } else {
                statusMessage = `Entrenando modelo...`;
            }

            updateActiveTrainingJob(targetTicker, targetModelType, {
                status: 'running',
                message: statusMessage,
                progress: progress
            });

            // Legacy support for current configuration
            if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
                setTrainingStatus('running');
                setPollingError(null);
                setTrainingStatusMessage(statusMessage);
            }

        } else if (status === 'queued') {
            // Still queued
            const statusMessage = responseMessage || `En cola...`;
            
            updateActiveTrainingJob(targetTicker, targetModelType, {
                status: 'queued',
                message: statusMessage
            });

            // Legacy support for current configuration
            if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
                setTrainingStatus('queued');
                setPollingError(null);
                setTrainingStatusMessage(statusMessage);
            }
        }
    }, [activeTrainingJobs, currentTrainingJob, config.selectedTicker, config.selectedModelType, 
        updateActiveTrainingJob, removeActiveTrainingJob, forecastToTrainFlow, forecastMutation]);// Cleanup polling on unmount
    useEffect(() => {
        return () => {
            if (pollingIntervalId) {
                clearInterval(pollingIntervalId);
            }
            // Clear job tracking state on unmount
            setIsPollingStatus(false);
            setCurrentJobId(null);
            setTrainingStatusMessage('');
            setPollingError(null);
            setForecastToTrainFlow(null);
        };
    }, [pollingIntervalId]);

    // --- Funciones de manejo de eventos ---
    const handleConfigChange = useCallback((newConfig) => {
        setConfig(prev => {
            const updatedConfig = { ...prev, ...newConfig };
            
            // Asegurarse que las fechas sean objetos Date si vienen del panel
            if (newConfig.startDate && !(newConfig.startDate instanceof Date)) {
                updatedConfig.startDate = new Date(newConfig.startDate);
            }
            if (newConfig.endDate && !(newConfig.endDate instanceof Date)) {
                 updatedConfig.endDate = new Date(newConfig.endDate);
            }
            
            // Manejar las nuevas fechas personalizadas
            if (newConfig.custom_start_date !== undefined) {
                if (newConfig.custom_start_date && !(newConfig.custom_start_date instanceof Date)) {
                    updatedConfig.custom_start_date = new Date(newConfig.custom_start_date);
                } else if (newConfig.custom_start_date === null) {
                    updatedConfig.custom_start_date = null;
                }
            }
            if (newConfig.custom_end_date !== undefined) {
                if (newConfig.custom_end_date && !(newConfig.custom_end_date instanceof Date)) {
                    updatedConfig.custom_end_date = new Date(newConfig.custom_end_date);
                } else if (newConfig.custom_end_date === null) {
                    updatedConfig.custom_end_date = null;
                }
            }
            
            // Manejar el preset de período de entrenamiento
            if (newConfig.training_period_preset !== undefined) {
                updatedConfig.training_period_preset = newConfig.training_period_preset;
            }
            
            return updatedConfig;        });
    }, []);    // Function to handle loading model data when clicking on trained models
    const handleLoadModelData = useCallback(async (modelType, modelName, extractedTicker) => {
        setIsLoadingModelData(true);
        setLoadedModelError(null);
        
        try {
            // Since getModelMetadata and getModelTrainingResults have been removed,
            // we'll use the available model metadata from availableModels instead
            const availableModelsData = availableModels[modelType];
            let selectedModel = null;
            
            if (availableModelsData && availableModelsData.models) {
                selectedModel = availableModelsData.models.find(model => 
                    (model.name || model.filename) === modelName
                );
            }

            // Parse the metadata to convert Python string literals to JavaScript objects
            const parsedMetadata = selectedModel ? parseMetadata(selectedModel.metadata) : null;

            // Create a simplified data structure using available metadata
            const loadedData = {
                id: `loaded-${modelType}-${extractedTicker}`,
                modelType: modelType.toUpperCase(),
                ticker: extractedTicker,
                modelName: modelName,
                metadata: parsedMetadata,
                isLoadedModel: true, // Flag to distinguish from current session results
                loadedAt: new Date().toISOString()
            };

            // Update the loaded model data state
            setLoadedModelData(loadedData);

            // Update last model used
            setLastModelUsed(`Modelo seleccionado: ${modelName} (${extractedTicker})`);            // Update training results to show in tabs - add loaded model to existing results
            const runId = `loaded-${modelType}-${extractedTicker}-${Date.now()}`;
            setTrainingResults(prevResults => ({
                ...prevResults,
                [runId]: {
                    id: runId,
                    modelType: modelType.toUpperCase(),
                    ticker: extractedTicker,
                    dateRange: parsedMetadata?.training_period || 'Datos del modelo guardado',
                    metrics: parsedMetadata?.metrics || {},
                    modelPath: selectedModel?.path || modelName,
                    timestamp: new Date().toISOString(),
                    bestParams: parsedMetadata?.best_params || {},
                    featureNames: parsedMetadata?.features_names || [],
                    isLoadedModel: true
                }
            }));

            message.success(`Modelo ${modelName} cargado exitosamente para ${extractedTicker}`);
            
        } catch (error) {
            console.error('Error loading model data:', error);
            setLoadedModelError(error.message);
            message.error(`Error al cargar datos del modelo: ${error.message}`);        } finally {
            setIsLoadingModelData(false);
        }    }, [availableModels]);

    // useEffect para resetear datos cuando cambia el ticker
    useEffect(() => {
        // Resetear datos si el ticker cambia
        setHistoricalData({ dates: [], values: [] });
        setForecastData([]);
        setLastModelUsed('');
        // Reset loaded model data when ticker changes
        setLoadedModelData(null);
        setLoadedModelError(null);
    }, [config.selectedTicker]); // Dependencia correcta

    useEffect(() => {
        const { startDate, endDate } = config;
        // Verificar si las fechas son válidas
        if (startDate instanceof Date && endDate instanceof Date && !isNaN(startDate) && !isNaN(endDate)) {
            if (endDate <= startDate) {
                setDateRangeWarning('Advertencia: La fecha de fin debe ser posterior a la fecha de inicio.');
                return;
            } 

            const diffTime = Math.abs(endDate - startDate); // Diferencia en milisegundos
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24)); // Convertir a días

            if (diffDays < min_calendar_days) {
                setDateRangeWarning(`Advertencia: El rango seleccionado (${diffDays} días) podría ser muy corto. Se recomiendan al menos ${min_calendar_days} días (~25 meses) para asegurar el cálculo de características.`);
            } else {
                setDateRangeWarning(''); // Limpiar advertencia si el rango es válido
            }
        } else {
            setDateRangeWarning('Advertencia: Las fechas seleccionadas no son válidas.');
        }
    }, [config.startDate, config.endDate]); // Dependencias para el efecto

    // useEffect para cargar modelos disponibles cuando cambia el tipo de modelo
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
                    setAvailableModelsError(null);
                })
                .catch(error => {
                    console.error(`Error loading available models for ${config.selectedModelType}:`, error);
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
    }, [config.selectedModelType]); // Dependencia en el tipo de modelo seleccionado    // El useCallback asegura que la función no se vuelva a crear en cada renderizado, lo que mejora el rendimiento
    const handleTrain = useCallback(async (modelType, currentConfig) => {
        // Clean up any existing polling before starting new training
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            setPollingIntervalId(null);
        }
        setIsPollingStatus(false);
        setCurrentJobId(null);
        setPollingError(null);
        setTrainingStatusMessage('');
        
        // Note: We don't clear forecastToTrainFlow here because this function
        // is also called from the automatic flow after forecast fails
        
        // Check if this specific configuration is already training
        const key = getTrainingKey(currentConfig.selectedTicker, modelType);
        const existingJob = activeTrainingJobs[key];
        
        if (existingJob && ['queued', 'running', 'submitting'].includes(existingJob.status)) {
            message.warning(`Ya hay un entrenamiento en progreso para ${currentConfig.selectedTicker} (${modelType.toUpperCase()}). Espera a que termine antes de iniciar otro.`);
            return Promise.reject(new Error('Training already in progress for this configuration'));
        }

        if (dateRangeWarning && dateRangeWarning.includes('podría ser muy corto')) {
            return new Promise((resolve, reject) => {
                Modal.confirm({
                    title: 'Advertencia sobre Rango de Fechas',
                    content: dateRangeWarning + "\n\nEl entrenamiento podría fallar. ¿Deseas continuar de todas formas?",
                    okText: 'Continuar',
                    cancelText: 'Cancelar',
                    onOk: () => {
                        // Solo si el usuario confirma, se ejecuta la mutación
                        setTrainingStatus('submitting');
                        setTrainingStatusMessage('Enviando solicitud de entrenamiento...');
                        trainMutation.mutateAsync({ modelType, config: currentConfig })
                            .then(resolve)
                            .catch(reject);
                    },
                    onCancel: () => {
                        message.info('Entrenamiento cancelado por el usuario.');
                        reject(new Error('Training cancelled by user'));
                    }
                });
            });
        } else if (dateRangeWarning && dateRangeWarning.includes('posterior a la fecha de inicio')) {
            message.error('Corrige las fechas antes de entrenar.');
            return Promise.reject(new Error('Invalid date range'));
        } else {
            // Si no hay advertencias o errores de fecha, entrena directamente
            setTrainingStatus('submitting');
            setTrainingStatusMessage('Enviando solicitud de entrenamiento...');
            return trainMutation.mutateAsync({ modelType, config: currentConfig });
        }
    }, [trainMutation, dateRangeWarning, pollingIntervalId, activeTrainingJobs, getTrainingKey]); // Updated dependencies
          // Maneja el pronóstico, similar a handleTrain pero para generar pronósticos
    const handleForecast = useCallback(async (modelType, currentConfig) => {
        // Clear any previous forecast-to-train flow state
        setForecastToTrainFlow(null);
        
        // Check if forecast is already in progress
        if (forecastMutation.isPending) {
            message.warning('Ya hay un pronóstico en progreso. Espera a que termine antes de iniciar otro.');
            return;
        }

        forecastMutation.mutate({ modelType, config: currentConfig });

    }, [forecastMutation]); // Dependencias para el useCallback

    const handleExportForecast = useCallback(() => {
        if (!forecastData || forecastData.length === 0) {
            message.warning('No hay datos de pronóstico para exportar.');
            return;
        }

        // Formatear datos para Papaparse (array de objetos está bien)
        const csvData = forecastData.map(item => ({
             Fecha: item.date,
             Prediccion: item.prediction?.toFixed(4) // Asegurar formato
         }));

        const csv = Papa.unparse(csvData);

        // Crear y descargar el archivo
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        const filename = `forecast_${config.selectedTicker}_${config.selectedModelType}_${dayjs().format('YYYYMMDD')}.csv`;
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    }, [forecastData, config.selectedTicker, config.selectedModelType]);

    // Encuentra el ID de la ejecución más reciente en trainingResults
    const latestRunId = Object.keys(trainingResults).length > 0
    ? Object.keys(trainingResults).sort((a, b) => dayjs(trainingResults[b].timestamp).unix() - dayjs(trainingResults[a].timestamp).unix())[0]
    : null;

    const currentMetricsToDisplay = latestRunId ? trainingResults[latestRunId].metrics : {};

    const latestRun = latestRunId ? trainingResults[latestRunId] : null;
    const currentBestParams = latestRun ? latestRun.bestParams : {};
    const featureNames = latestRun?.featureNames || [];

    const [residualsData, setResidualsData] = useState({ 
        dates: [], 
        values: [],
        acf: null,
        pacf: null
    });

    // --- Define los items para las pestañas ---
    const resultTabs = [
        {
            key: '1', 
            label: 'Gráfico y Pronóstico', 
            children: ( // Contenido de la pestaña
                <div> {/* Envuelve en un div para añadir el botón */}
                    <Button
                        onClick={handleExportForecast}
                        disabled={forecastData.length === 0}
                        style={{ marginBottom: '15px' }}
                    >
                        Exportar Pronóstico a CSV
                    </Button>
                    <GraphDisplay
                        historicalData={historicalData}
                        forecastData={forecastData}
                        ticker={config.selectedTicker}
                    />
                </div>
            ),
        },
        {
            key: '2',
            label: 'Métricas Detalladas',
            children: (
                <MetricsDisplay metrics={currentMetricsToDisplay} />
            ),
             disabled: !latestRunId || Object.keys(currentMetricsToDisplay).length === 0
        },
        {
            key: '3',
            label: 'Detalles del Modelo',
            children: <ModelDetailsDisplay latestRun={latestRun} />,
            disabled: !latestRunId
        },
        {
            key: '4',
            label: 'Comparación de Modelos',
            children: <ModelComparisonTable results={trainingResults} />,
            disabled: Object.keys(trainingResults).length === 0 // Deshabilitar si no hay resultados
        },
        {
            key: '5',
            label: 'Residuales',
            children: (
                <ResidualsDisplay 
                data={residualsData}
                />
            ),
            disabled: !residualsData || residualsData.values.length === 0
        },
        {
            key: '6',
            label: '🔬 Interpretabilidad SHAP',
            children: (
                <ShapExplainer 
                    ticker={config.selectedTicker}
                    onError={(err) => message.error(`Error SHAP: ${err.message}`)}
                />
            ),
            disabled: false // SHAP siempre disponible para XGBoost y RF
        },
        {
            key: '7',
            label: '🎯 Predicción Ensemble',
            children: (
                <EnsemblePredictor 
                    ticker={config.selectedTicker}
                    onPrediction={(result) => {
                        message.success(`Predicción ensemble generada para ${config.selectedTicker}`);
                        console.log('Ensemble prediction result:', result);
                    }}
                    onError={(err) => message.error(`Error Ensemble: ${err.message}`)}
                />
            ),
            disabled: false // Ensemble siempre disponible
        }
    ];


    // --- Renderizado con Tabs ---
    return (
        <ConfigProvider
             theme={{
                 // Algoritmo para tema claro u oscuro
                 algorithm: currentTheme === 'dark' ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
                 // Puedes personalizar tokens aquí si quieres
                 // token: { colorPrimary: '#00b96b' },
             }}
         >
            <Layout style={{ minHeight: '100vh' }}>
                <Header style={{ color: 'white', textAlign: 'center' }}>
                <h1 style={{ margin: 0, fontSize: '1.8em' }}> {/* Ajusta fontSize si es necesario */}
                        Pronóstico de Series de Tiempo Financieras
                </h1>
                <Switch
                    checkedChildren={<MoonOutlined />}
                    unCheckedChildren={<SunOutlined />}
                    onChange={toggleTheme}
                    checked={currentTheme === 'dark'}
                    style={{ position: 'absolute', right: '20px', top: '22px' }} // Posicionar el switch
                />
                <Button
                    type="link"
                    onClick={() => setHelpModalVisible(true)}
                    style={{ position: 'absolute', right: '100px', top: '16px' }} // Posicionar el botón
                >
                    Ayuda
                </Button>
                <HelpModal
                    visible={helpModalVisible}
                    onClose={() => setHelpModalVisible(false)}
                />
                </Header>
                <Content style={{ padding: '20px ' }}>
                    <div style={{ padding: '0 50px' }}>                    {/* Mostrar error del hook si existe */}
                    {(trainMutation.error || forecastMutation.error) && (
                        <Alert
                            message="Error"
                            description={trainMutation.error?.message || forecastMutation.error?.message || 'Ocurrió un error'}
                            type="error"
                            showIcon
                            closable
                            // Opcional: resetear el estado de error del hook al cerrar
                            // onClose={() => { trainMutation.reset(); forecastMutation.reset(); }}
                        />
                    )}
                    
                    {/* Active Training Jobs Display */}
                    {getActiveTrainingJobsCount() > 0 && (
                        <ActiveTrainingJobs
                            activeJobs={getActiveTrainingJobsForDisplay()}
                            onCancelJob={(job) => {
                                removeActiveTrainingJob(job.ticker, job.modelType);
                                message.info(`Entrenamiento de ${job.modelType.toUpperCase()} para ${job.ticker} eliminado de la lista.`);
                            }}
                            onRetryJob={(job) => {
                                if (job.status === 'failed') {
                                    message.info(`Reintentando entrenamiento de ${job.modelType.toUpperCase()} para ${job.ticker}...`);
                                    handleTrain(job.modelType, job.config);
                                    removeActiveTrainingJob(job.ticker, job.modelType);
                                }
                            }}
                            style={{ marginBottom: '16px' }}
                        />
                    )}
                    
                    <Row gutter={[16, 16]}>                        <Col xs={24} md={10} lg={8} xl={7}>
                            {/* Show spinner only for active operations */}
                            <Spin spinning={forecastMutation.isPending || trainMutation.isPending} 
                                  tip={
                                    forecastMutation.isPending ? "Generando pronóstico..." :
                                    trainMutation.isPending ? "Enviando entrenamiento..." :
                                    "Procesando..."
                                  }>
                                <Card title={
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span>Configuración</span>
                                        
                                        {/* Enhanced status display */}
                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                                            {/* Show active jobs count if any */}
                                            {getActiveTrainingJobsCount() > 0 && (
                                                <div style={{ 
                                                    fontSize: '11px', 
                                                    color: '#1890ff',
                                                    marginBottom: '2px'
                                                }}>
                                                    🚀 {getActiveTrainingJobsCount()} entrenamiento{getActiveTrainingJobsCount() > 1 ? 's' : ''} activo{getActiveTrainingJobsCount() > 1 ? 's' : ''}
                                                </div>
                                            )}
                                            
                                            {/* Show status for current configuration only if relevant */}
                                            {(isCurrentConfigurationTraining() || trainingStatusMessage) && (
                                                <>
                                                    <span style={{ 
                                                        fontSize: '12px', 
                                                        color: 
                                                            trainingStatus === 'completed' ? '#52c41a' :
                                                            trainingStatus === 'failed' ? '#ff4d4f' :
                                                            trainingStatus === 'running' ? '#1890ff' :
                                                            '#faad14'
                                                    }}>
                                                        {trainingStatus === 'queued' ? `En cola${currentJobId ? ` (ID: ${currentJobId.slice(0,8)}...)` : ''}` :
                                                         trainingStatus === 'running' ? `Entrenando${currentJobId ? ` (ID: ${currentJobId.slice(0,8)}...)` : ''}` :
                                                         trainingStatus === 'completed' ? 'Completado' :
                                                         trainingStatus === 'failed' ? 'Falló' :
                                                         trainingStatus === 'submitting' ? 'Enviando...' :
                                                         trainingStatus}
                                                    </span>
                                                    {trainingStatusMessage && (
                                                        <span style={{ 
                                                            fontSize: '10px', 
                                                            color: '#666',
                                                            marginTop: '2px',
                                                            maxWidth: '200px',
                                                            textAlign: 'right'
                                                        }}>
                                                            {trainingStatusMessage}
                                                        </span>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    </div>
                                }><ConfigurationPanel // La versión AntD
                                        availableModelTypes={availableModelTypes}
                                        availableTickers={availableTickers}
                                        onConfigChange={handleConfigChange}
                                        onTrain={handleTrain} // Pasas la función handleTrain de App
                                        onForecast={handleForecast} // Pasas la función handleForecast de App
                                        initialConfig={config}
                                        // ---- PASAR LA ADVERTENCIA COMO PROP ----
                                        dateRangeWarning={dateRangeWarning}                                        // ---- PASAR EL ESTADO DEL ENTRENAMIENTO ----                                        trainingStatus={trainingStatus}
                                        currentTrainingJob={currentTrainingJob}
                                        currentJobId={currentJobId}
                                        isPollingStatus={isPollingStatus}
                                        trainingStatusMessage={trainingStatusMessage}
                                        // ---- PASAR LOS ESTADOS DE CARGA ----
                                        trainMutationPending={trainMutation.isPending}
                                        forecastMutationPending={forecastMutation.isPending}
                                        pollingError={pollingError}                                        // ---- NEW MULTI-JOB PROPS ----
                                        isCurrentConfigurationTraining={isCurrentConfigurationTraining()}
                                        activeTrainingJobs={getActiveTrainingJobsForDisplay()}                                        // ---- AVAILABLE MODELS PROPS ----
                                        availableModels={availableModels[config.selectedModelType] || null}
                                        availableModelsLoading={availableModelsLoading}
                                        availableModelsError={availableModelsError}
                                        // ---- MODEL DATA LOADING PROPS ----
                                        onLoadModelData={handleLoadModelData}
                                        isLoadingModelData={isLoadingModelData}
                                        loadedModelError={loadedModelError}
                                        // --------------------------------------
                                    />
                                </Card>
                            </Spin>
                        </Col>
                        <Col xs={24} md={14} lg={16} xl={17}>
                            <Card title={`Resultados ${lastModelUsed ? `(${lastModelUsed})` : ''}`}>
                                <Tabs defaultActiveKey="1" items={resultTabs} />
                            </Card>
                        </Col>
                    </Row>
                    </div>
                </Content>
                <Footer style={{ textAlign: 'center' }}>
                    Series de Tiempo Financieras V.0.1 ©{new Date().getFullYear()} Creado con Ant Design
                </Footer>
            </Layout>
        </ConfigProvider>
    );
}

export default App;