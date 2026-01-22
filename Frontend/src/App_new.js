/**
 * App.js - Componente principal refactorizado
 * StockTimePredictor - Aplicación de predicción de series financieras
 * 
 * Arquitectura mejorada con:
 * - Gestión de estado centralizada (AppContext)
 * - Componentes modulares y reutilizables
 * - Sistema de temas profesional
 * - UX mejorada con feedback visual
 */
import React, { useCallback, useEffect, useState } from 'react';
import dayjs from 'dayjs';
import { 
  Layout, 
  Row, 
  Col, 
  Card, 
  Spin, 
  Alert, 
  Tabs, 
  message, 
  Button, 
  ConfigProvider, 
  theme as antdTheme, 
  Modal 
} from 'antd';
import Papa from 'papaparse';

// Context
import { AppProvider, useAppContext } from './context/AppContext';

// Theme
import { lightTheme, darkTheme } from './theme/themeConfig';

// Layout Components
import AppHeader from './components/layout/AppHeader';

// Dashboard Components
import StatsOverview from './components/dashboard/StatsOverview';
import QuickActions from './components/dashboard/QuickActions';

// Form Components  
import TickerSelector from './components/forms/TickerSelector';
import ModelSelector from './components/forms/ModelSelector';

// Display Components
import ResidualsDisplay from './components/ResidualsDisplay';
import ConfigurationPanel from './components/ConfigurationPanel_AntD';
import GraphDisplay from './components/GraphDisplay';
import MetricsDisplay from './components/MetricsDisplay_AntD';
import ModelComparisonTable from './components/ModelComparisonTable';
import ModelDetailsDisplay from './components/ModelDetailsDisplay';
import HelpModal from './components/HelpModal';
import ActiveTrainingJobs from './components/ActiveTrainingJobs';

// Hooks
import { 
  useTrainModelMutation, 
  useGenerateForecastMutation, 
  useTrainingStatusMutation 
} from './hooks/useApiMutations';

// Services
import { getAvailableModels } from './services/api';

// Utils
import { parseMetadata } from './utils/pythonUtils';

// Styles
import './styles/globals.css';

const { Content, Footer } = Layout;

// ========================================
// COMPONENTE PRINCIPAL DEL DASHBOARD
// ========================================
function Dashboard() {
  const {
    // Configuración
    config,
    handleConfigChange,
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
    availableTickers,
    availableModelTypes,
    pollingTimeoutMs,
    
    // Helpers
    getTrainingKey,
  } = useAppContext();

  // Estado local para el flujo de forecast-to-train
  const [forecastToTrainFlow, setForecastToTrainFlow] = useState(null);
  
  // Estado legacy para compatibilidad
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [currentTrainingJob, setCurrentTrainingJob] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [isPollingStatus, setIsPollingStatus] = useState(false);
  const [trainingStatusMessage, setTrainingStatusMessage] = useState('');
  const [pollingError, setPollingError] = useState(null);
  const [pollingIntervalId, setPollingIntervalId] = useState(null);

  // ========================================
  // HOOKS DE MUTACIÓN
  // ========================================
  
  const trainMutation = useTrainModelMutation({
    onSuccessCallback: (jobData, variables) => {
      const { job_id } = jobData;
      const modelType = variables.modelType;
      const jobConfig = variables.config;
      const ticker = jobConfig.selectedTicker;
      
      // Agregar a trabajos activos
      addActiveTrainingJob(ticker, modelType, {
        jobId: job_id,
        status: 'queued',
        message: `Entrenamiento iniciado (ID: ${job_id.slice(0, 8)}...)`,
        config: jobConfig
      });
      
      // Soporte legacy
      if (ticker === config.selectedTicker && modelType === config.selectedModelType) {
        setCurrentTrainingJob({
          jobId: job_id,
          modelType: modelType,
          config: jobConfig,
          startTime: new Date()
        });
        setCurrentJobId(job_id);
        setIsPollingStatus(true);
        setTrainingStatus('queued');
        setTrainingStatusMessage(`Entrenamiento iniciado con ID: ${job_id.slice(0, 8)}...`);
      }
      
      // Iniciar polling
      startStatusPollingForJob(ticker, modelType, job_id, 5);
    },
    onErrorCallback: (err, variables) => {
      const modelType = variables?.modelType;
      const ticker = variables?.config?.selectedTicker;
      
      if (ticker && modelType) {
        removeActiveTrainingJob(ticker, modelType);
      }
      
      message.error(`Error al iniciar entrenamiento: ${err.message}`);
    }
  });

  const forecastMutation = useGenerateForecastMutation({
    onSuccessCallback: (forecastResult) => {
      setHistoricalData({
        dates: forecastResult.historical_dates || [],
        values: forecastResult.historical_values || []
      });
      setForecastData(forecastResult.predictions || []);
      setLastModelUsed(forecastResult.model_used || `Predicted with ${config.selectedModelType.toUpperCase()}`);
      message.success('Pronóstico generado exitosamente');
    },
    onErrorCallback: (err, variables) => {
      setHistoricalData({ dates: [], values: [] });
      setForecastData([]);
      
      const isModelNotFoundError = err?.response?.status === 404 || 
        err?.message?.includes('modelo no encontrado');
      
      if (isModelNotFoundError && variables) {
        const { modelType, config: forecastConfig } = variables;
        const ticker = forecastConfig.selectedTicker;
        
        setForecastToTrainFlow({ modelType, config: forecastConfig });
        
        Modal.confirm({
          title: 'Modelo no encontrado',
          content: `No se encontró un modelo entrenado de tipo ${modelType.toUpperCase()} para ${ticker}. ¿Deseas entrenar un modelo ahora?`,
          okText: 'Entrenar ahora',
          cancelText: 'Cancelar',
          onOk: () => {
            message.info(`Iniciando entrenamiento...`);
            handleTrain(modelType, forecastConfig);
            setForecastToTrainFlow(null);
          },
          onCancel: () => {
            setForecastToTrainFlow(null);
          }
        });
      }
    }
  });

  const statusMutation = useTrainingStatusMutation({
    onSuccessCallback: (statusData) => {
      handleTrainingStatusUpdate(statusData);
    },
    onErrorCallback: (error) => {
      console.error('Error checking training status:', error);
      Object.keys(pollingIntervals).forEach(key => {
        clearInterval(pollingIntervals[key]);
      });
      setPollingIntervals({});
      message.error('Error consultando el estado del entrenamiento');
    }
  });

  // ========================================
  // FUNCIONES DE POLLING
  // ========================================
  
  const startStatusPollingForJob = useCallback((ticker, modelType, jobId, intervalSeconds = 5) => {
    const key = getTrainingKey(ticker, modelType);
    const validInterval = Math.max(5, Math.min(10, intervalSeconds));
    const intervalMs = validInterval * 1000;

    if (pollingIntervals[key]) {
      clearInterval(pollingIntervals[key]);
    }

    const intervalId = setInterval(() => {
      statusMutation.mutate({ modelType, jobId });
    }, intervalMs);

    setPollingIntervals(prev => ({
      ...prev,
      [key]: intervalId
    }));
    
    // Timeout de 10 minutos
    setTimeout(() => {
      if (pollingIntervals[key] === intervalId) {
        stopPollingForJob(ticker, modelType);
        updateActiveTrainingJob(ticker, modelType, {
          status: 'timeout',
          message: 'Tiempo de espera agotado'
        });
        message.warning('El tiempo de espera del entrenamiento ha expirado.');
      }
    }, pollingTimeoutMs);
  }, [statusMutation, pollingIntervals, getTrainingKey, pollingTimeoutMs, updateActiveTrainingJob]);

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
  }, [pollingIntervals, getTrainingKey]);

  // ========================================
  // MANEJO DE ACTUALIZACIONES DE ESTADO
  // ========================================
  
  const handleTrainingStatusUpdate = useCallback((statusData) => {
    const { status, result, error, progress, message: responseMessage, job_id } = statusData;

    // Encontrar el trabajo correspondiente
    let targetJob = null;
    let targetTicker = null;
    let targetModelType = null;

    if (job_id) {
      Object.values(activeTrainingJobs).forEach(job => {
        if (job.jobId === job_id) {
          targetJob = job;
          targetTicker = job.ticker;
          targetModelType = job.modelType;
        }
      });
    }

    if (!targetJob && currentTrainingJob) {
      targetJob = currentTrainingJob;
      targetTicker = currentTrainingJob.config?.selectedTicker;
      targetModelType = currentTrainingJob.modelType;
    }

    if (!targetTicker || !targetModelType) return;

    if (status === 'SUCCESS' || status === 'completed') {
      if (targetJob && result) {
        const startDate = targetJob.config.startDate.toISOString().split('T')[0];
        const endDate = targetJob.config.endDate.toISOString().split('T')[0];
        const runId = `${targetModelType}-${targetTicker}-${startDate}-${endDate}`;

        updateActiveTrainingJob(targetTicker, targetModelType, {
          status: 'completed',
          message: 'Entrenamiento completado exitosamente',
          result: result
        });

        if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
          setResidualsData({
            dates: result.residual_dates || [],
            values: result.residuals || [],
            acf: result.acf || null,
            pacf: result.pacf || null
          });

          if (result.historical_dates && result.historical_values) {
            setHistoricalData({
              dates: result.historical_dates,
              values: result.historical_values
            });
          }

          setLastModelUsed(result.model_path || `Trained ${targetModelType.toUpperCase()}`);
          setTrainingStatus('completed');
          setCurrentTrainingJob(null);
          setTrainingStatusMessage('');
        }

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

        // Auto-generar pronóstico si venimos del flujo forecast-to-train
        if (forecastToTrainFlow && forecastToTrainFlow.modelType === targetModelType) {
          const { config: forecastConfig } = forecastToTrainFlow;
          message.success(`¡Modelo entrenado! Generando pronóstico automáticamente...`);
          setForecastToTrainFlow(null);
          setTimeout(() => {
            forecastMutation.mutate({ modelType: targetModelType, config: forecastConfig });
          }, 1000);
        } else {
          message.success(`¡Modelo ${targetModelType.toUpperCase()} para ${targetTicker} entrenado exitosamente!`);
        }

        stopPollingForJob(targetTicker, targetModelType);
        setTimeout(() => {
          removeActiveTrainingJob(targetTicker, targetModelType);
        }, 3000);
      }
    } else if (status === 'FAILURE' || status === 'failed') {
      const errorMessage = error || responseMessage || 'Error desconocido';
      
      updateActiveTrainingJob(targetTicker, targetModelType, {
        status: 'failed',
        message: `Error: ${errorMessage}`,
        error: errorMessage
      });

      if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
        setTrainingStatus('failed');
        setCurrentTrainingJob(null);
        setPollingError(errorMessage);
        setTrainingStatusMessage('');
      }
      
      if (forecastToTrainFlow && forecastToTrainFlow.modelType === targetModelType) {
        setForecastToTrainFlow(null);
      }
      
      message.error(`Entrenamiento falló: ${errorMessage}`);
      stopPollingForJob(targetTicker, targetModelType);
    } else if (status === 'PROGRESS' || status === 'running' || status === 'PENDING') {
      let statusMessage = progress ? `Progreso: ${progress}%` : responseMessage || 'Entrenando modelo...';

      updateActiveTrainingJob(targetTicker, targetModelType, {
        status: 'running',
        message: statusMessage,
        progress: progress
      });

      if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
        setTrainingStatus('running');
        setPollingError(null);
        setTrainingStatusMessage(statusMessage);
      }
    } else if (status === 'queued') {
      const statusMessage = responseMessage || 'En cola...';
      
      updateActiveTrainingJob(targetTicker, targetModelType, {
        status: 'queued',
        message: statusMessage
      });

      if (targetTicker === config.selectedTicker && targetModelType === config.selectedModelType) {
        setTrainingStatus('queued');
        setPollingError(null);
        setTrainingStatusMessage(statusMessage);
      }
    }
  }, [
    activeTrainingJobs, currentTrainingJob, config, forecastToTrainFlow,
    updateActiveTrainingJob, removeActiveTrainingJob, stopPollingForJob,
    setHistoricalData, setResidualsData, setLastModelUsed, setTrainingResults,
    forecastMutation
  ]);

  // ========================================
  // HANDLERS PRINCIPALES
  // ========================================
  
  const handleTrain = useCallback(async (modelType, currentConfig) => {
    if (pollingIntervalId) {
      clearInterval(pollingIntervalId);
      setPollingIntervalId(null);
    }
    setIsPollingStatus(false);
    setCurrentJobId(null);
    setPollingError(null);
    setTrainingStatusMessage('');
    
    const key = getTrainingKey(currentConfig.selectedTicker, modelType);
    const existingJob = activeTrainingJobs[key];
    
    if (existingJob && ['queued', 'running', 'submitting'].includes(existingJob.status)) {
      message.warning(`Ya hay un entrenamiento en progreso para ${currentConfig.selectedTicker} (${modelType.toUpperCase()}).`);
      return Promise.reject(new Error('Training already in progress'));
    }

    if (dateRangeWarning && dateRangeWarning.includes('podría ser muy corto')) {
      return new Promise((resolve, reject) => {
        Modal.confirm({
          title: 'Advertencia sobre Rango de Fechas',
          content: dateRangeWarning + "\n\n¿Deseas continuar?",
          okText: 'Continuar',
          cancelText: 'Cancelar',
          onOk: () => {
            setTrainingStatus('submitting');
            setTrainingStatusMessage('Enviando solicitud...');
            trainMutation.mutateAsync({ modelType, config: currentConfig }).then(resolve).catch(reject);
          },
          onCancel: () => reject(new Error('Cancelled'))
        });
      });
    }
    
    setTrainingStatus('submitting');
    setTrainingStatusMessage('Enviando solicitud...');
    return trainMutation.mutateAsync({ modelType, config: currentConfig });
  }, [trainMutation, dateRangeWarning, pollingIntervalId, activeTrainingJobs, getTrainingKey]);

  const handleForecast = useCallback(async (modelType, currentConfig) => {
    setForecastToTrainFlow(null);
    
    if (forecastMutation.isPending) {
      message.warning('Ya hay un pronóstico en progreso.');
      return;
    }

    forecastMutation.mutate({ modelType, config: currentConfig });
  }, [forecastMutation]);

  const handleExportForecast = useCallback(() => {
    if (!forecastData || forecastData.length === 0) {
      message.warning('No hay datos de pronóstico para exportar.');
      return;
    }

    const csvData = forecastData.map(item => ({
      Fecha: item.date,
      Prediccion: item.prediction?.toFixed(4)
    }));

    const csv = Papa.unparse(csvData);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    const filename = `forecast_${config.selectedTicker}_${config.selectedModelType}_${dayjs().format('YYYYMMDD')}.csv`;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    message.success('Pronóstico exportado exitosamente');
  }, [forecastData, config]);

  const handleLoadModelData = useCallback(async (modelType, modelName, extractedTicker) => {
    setIsLoadingModelData(true);
    setLoadedModelError(null);
    
    try {
      const availableModelsData = availableModels[modelType];
      let selectedModel = null;
      
      if (availableModelsData && availableModelsData.models) {
        selectedModel = availableModelsData.models.find(model => 
          (model.name || model.filename) === modelName
        );
      }

      const parsedMetadata = selectedModel ? parseMetadata(selectedModel.metadata) : null;

      const loadedData = {
        id: `loaded-${modelType}-${extractedTicker}`,
        modelType: modelType.toUpperCase(),
        ticker: extractedTicker,
        modelName: modelName,
        metadata: parsedMetadata,
        isLoadedModel: true,
        loadedAt: new Date().toISOString()
      };

      setLoadedModelData(loadedData);
      setLastModelUsed(`Modelo seleccionado: ${modelName} (${extractedTicker})`);

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

      message.success(`Modelo ${modelName} cargado exitosamente`);
    } catch (error) {
      setLoadedModelError(error.message);
      message.error(`Error al cargar modelo: ${error.message}`);
    } finally {
      setIsLoadingModelData(false);
    }
  }, [availableModels, setLoadedModelData, setIsLoadingModelData, setLoadedModelError, setLastModelUsed, setTrainingResults]);

  const handleRefreshModels = useCallback(() => {
    if (config.selectedModelType) {
      getAvailableModels(config.selectedModelType)
        .then(response => {
          setAvailableModels(prev => ({
            ...prev,
            [config.selectedModelType]: response
          }));
          message.success('Modelos actualizados');
        })
        .catch(error => {
          message.error('Error al actualizar modelos');
        });
    }
  }, [config.selectedModelType, setAvailableModels]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
      }
    };
  }, [pollingIntervalId]);

  // ========================================
  // DATOS PARA RENDERIZADO
  // ========================================
  
  const latestRunId = Object.keys(trainingResults).length > 0
    ? Object.keys(trainingResults).sort((a, b) => 
        dayjs(trainingResults[b].timestamp).unix() - dayjs(trainingResults[a].timestamp).unix()
      )[0]
    : null;

  const currentMetricsToDisplay = latestRunId ? trainingResults[latestRunId].metrics : {};
  const latestRun = latestRunId ? trainingResults[latestRunId] : null;

  // Tabs de resultados
  const resultTabs = [
    {
      key: '1',
      label: '📈 Gráfico y Pronóstico',
      children: (
        <div>
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
      label: '📊 Métricas',
      children: <MetricsDisplay metrics={currentMetricsToDisplay} />,
      disabled: !latestRunId || Object.keys(currentMetricsToDisplay).length === 0
    },
    {
      key: '3',
      label: '🔧 Detalles del Modelo',
      children: <ModelDetailsDisplay latestRun={latestRun} />,
      disabled: !latestRunId
    },
    {
      key: '4',
      label: '📋 Comparación',
      children: <ModelComparisonTable results={trainingResults} />,
      disabled: Object.keys(trainingResults).length === 0
    },
    {
      key: '5',
      label: '📉 Residuales',
      children: <ResidualsDisplay data={residualsData} />,
      disabled: !residualsData || residualsData.values.length === 0
    }
  ];

  // ========================================
  // RENDERIZADO
  // ========================================
  
  return (
    <ConfigProvider
      theme={{
        algorithm: currentTheme === 'dark' ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        ...( currentTheme === 'dark' ? darkTheme : lightTheme)
      }}
    >
      <Layout className={`app-container ${currentTheme === 'dark' ? 'dark-mode' : ''}`}>
        {/* Header */}
        <AppHeader
          currentTheme={currentTheme}
          onThemeToggle={toggleTheme}
          onHelpClick={() => setHelpModalVisible(true)}
          activeJobsCount={getActiveTrainingJobsCount()}
        />

        <Content className="app-content">
          {/* Stats Overview */}
          <StatsOverview
            selectedTicker={config.selectedTicker}
            selectedModelType={config.selectedModelType}
            activeJobsCount={getActiveTrainingJobsCount()}
            forecastHorizon={config.forecastHorizon}
          />

          {/* Quick Actions */}
          <QuickActions
            onForecast={() => handleForecast(config.selectedModelType, config)}
            onTrain={(modelType) => handleTrain(modelType || config.selectedModelType, config)}
            onExport={handleExportForecast}
            onRefreshModels={handleRefreshModels}
            isTraining={trainMutation.isPending || isCurrentConfigurationTraining()}
            isForecastPending={forecastMutation.isPending}
            selectedTicker={config.selectedTicker}
            selectedModelType={config.selectedModelType}
            hasForecastData={forecastData.length > 0}
          />

          {/* Errores */}
          {(trainMutation.error || forecastMutation.error) && (
            <Alert
              message="Error"
              description={trainMutation.error?.message || forecastMutation.error?.message}
              type="error"
              showIcon
              closable
              style={{ marginBottom: '24px' }}
            />
          )}

          {/* Active Training Jobs */}
          {getActiveTrainingJobsCount() > 0 && (
            <ActiveTrainingJobs
              activeJobs={getActiveTrainingJobsForDisplay()}
              onCancelJob={(job) => {
                removeActiveTrainingJob(job.ticker, job.modelType);
                message.info(`Entrenamiento eliminado de la lista.`);
              }}
              onRetryJob={(job) => {
                if (job.status === 'failed') {
                  handleTrain(job.modelType, job.config);
                  removeActiveTrainingJob(job.ticker, job.modelType);
                }
              }}
              style={{ marginBottom: '24px' }}
            />
          )}

          {/* Main Content */}
          <Row gutter={[24, 24]}>
            {/* Configuration Panel */}
            <Col xs={24} lg={8} xl={7}>
              <Spin spinning={forecastMutation.isPending || trainMutation.isPending}>
                <Card 
                  title={
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span>⚙️ Configuración</span>
                      {getActiveTrainingJobsCount() > 0 && (
                        <span style={{ fontSize: '11px', color: '#1890ff' }}>
                          ({getActiveTrainingJobsCount()} activo{getActiveTrainingJobsCount() > 1 ? 's' : ''})
                        </span>
                      )}
                    </div>
                  }
                  className="config-panel"
                >
                  <ConfigurationPanel
                    availableModelTypes={availableModelTypes}
                    availableTickers={availableTickers}
                    onConfigChange={handleConfigChange}
                    onTrain={handleTrain}
                    onForecast={handleForecast}
                    initialConfig={config}
                    dateRangeWarning={dateRangeWarning}
                    trainingStatus={trainingStatus}
                    currentTrainingJob={currentTrainingJob}
                    currentJobId={currentJobId}
                    isPollingStatus={isPollingStatus}
                    trainingStatusMessage={trainingStatusMessage}
                    trainMutationPending={trainMutation.isPending}
                    forecastMutationPending={forecastMutation.isPending}
                    pollingError={pollingError}
                    isCurrentConfigurationTraining={isCurrentConfigurationTraining()}
                    activeTrainingJobs={getActiveTrainingJobsForDisplay()}
                    availableModels={availableModels[config.selectedModelType] || null}
                    availableModelsLoading={availableModelsLoading}
                    availableModelsError={availableModelsError}
                    onLoadModelData={handleLoadModelData}
                    isLoadingModelData={isLoadingModelData}
                    loadedModelError={loadedModelError}
                  />
                </Card>
              </Spin>
            </Col>

            {/* Results Section */}
            <Col xs={24} lg={16} xl={17}>
              <Card 
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>📊 Resultados {lastModelUsed ? `(${lastModelUsed})` : ''}</span>
                    {latestRun && (
                      <span style={{ 
                        fontSize: '12px', 
                        padding: '4px 12px', 
                        background: '#ebf8ff', 
                        borderRadius: '20px',
                        color: '#2c5282'
                      }}>
                        {latestRun.modelType} • {latestRun.ticker}
                      </span>
                    )}
                  </div>
                }
                className="results-section"
              >
                <Tabs defaultActiveKey="1" items={resultTabs} />
              </Card>
            </Col>
          </Row>
        </Content>

        <Footer className="app-footer">
          <div>
            StockTime Predictor v1.0 ©{new Date().getFullYear()} • 
            <span style={{ marginLeft: '8px' }}>
              Creado con React, Ant Design & FastAPI
            </span>
          </div>
        </Footer>

        {/* Help Modal */}
        <HelpModal
          visible={helpModalVisible}
          onClose={() => setHelpModalVisible(false)}
        />
      </Layout>
    </ConfigProvider>
  );
}

// ========================================
// APP WRAPPER CON CONTEXT PROVIDER
// ========================================
function App() {
  return (
    <AppProvider>
      <Dashboard />
    </AppProvider>
  );
}

export default App;
