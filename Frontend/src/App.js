import React, { useState, useCallback, useEffect } from 'react';
import dayjs from 'dayjs';
import { Layout, Row, Col, Card, Spin, Alert, Tabs, message, Button, Switch, ConfigProvider, theme as antdTheme, Descriptions  } from 'antd';
import { SunOutlined, MoonOutlined } from '@ant-design/icons';
import Papa from 'papaparse';
import ConfigurationPanel from './components/ConfigurationPanel';
import GraphDisplay from './components/GraphDisplay';
import MetricsDisplay from './components/MetricsDisplay';
import ModelComparisonTable from './components/ModelComparisonTable';
import ModelDetailsDisplay from './components/ModelDetailsDisplay';
import { useTrainModelMutation, useGenerateForecastMutation } from './hooks/useApiMutations'; // Ajusta el path
import './App.css';

// Extraer componentes de Layout para claridad
const { Header, Content, Footer } = Layout;

const min_calendar_days = 760; // Mínimo de días calendario para el rango de fechas

/**
 * Main App component that manages the configuration, data, and processes related to financial forecasting and model training.
 *
 * It includes states, mutation hooks, and utility functions necessary for:
 * - Configuring the forecasting model parameters.
 * - Handling historical and forecast data.
 * - Training Machine Learning models.
 * - Generating forecasts based on the selected model.
 * - Exporting forecast results.
 *
 * The component utilizes useState, useEffect, and useCallback hooks for state management and performance optimization.
 * - `trainMutation`: For training machine learning models.
 * - `forecastMutation`: For generating forecasts.
 * - `handleConfigChange`: For managing configuration changes.
 * - `handleTrain` and `handleForecast`: For executing training and forecasting processes.
 * - `handleExportForecast`: For exporting forecasted results to CSV.
 *
 * @return {React.Element} The App component responsible for managing states and rendering the application's main functionality.
 */
function App() {
    const [config, setConfig] = useState({
        selectedModelType: 'rf',
        selectedTicker: 'NU',
        startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 3)), // 3 años atrás
        endDate: new Date(),
        forecastHorizon: 10,
        nLags: 10,
        targetCol: 'Close'
    });
    const [historicalData, setHistoricalData] = useState({ dates: [], values: [] });
    const [forecastData, setForecastData] = useState([]);
    const [trainingResults, setTrainingResults] = useState({});
    const [lastModelUsed, setLastModelUsed] = useState('');
    const [availableModelTypes] = useState(['rf', 'lstm', 'xgboost']);
    const [availableTickers] = useState(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NU']);
    const [dateRangeWarning, setDateRangeWarning] = useState('');
    const [currentTheme, setCurrentTheme] = useState('light');
    const toggleTheme = (checked) => {
        setCurrentTheme(checked ? 'dark' : 'light');
    };

     // --- Usar los hooks de mutación ---
    const trainMutation = useTrainModelMutation({
        onSuccessCallback: (result, variables) => { // Callback para actualizar el estado de App
            const modelType = variables.modelType;
            const ticker = variables.config.selectedTicker;
            const startDate = variables.config.startDate.toISOString().split('T')[0];
            const endDate = variables.config.endDate.toISOString().split('T')[0];
            const runId = `<span class="math-inline">\{modelType\}\-</span>{ticker}-<span class="math-inline">\{startDate\}\-</span>{endDate}`;

            setTrainingResults(prevResults => ({
                ...prevResults,
                [runId]: { // Usar runId como clave
                    id: runId,
                    modelType: modelType.toUpperCase(),
                    ticker: ticker,
                    dateRange: `${startDate} / ${endDate}`,
                    metrics: result.metrics || {},
                    modelPath: result.model_path || 'N/A',
                    timestamp: new Date().toISOString(),
                    bestParams: result.best_params || {},
                    featureNames: result.features_names || []
                }
            }));
            setLastModelUsed(result.model_path || `Trained ${modelType.toUpperCase()}`);
        },
        onErrorCallback: (err) => {
            // Puedes mantener el setError si quieres mostrarlo en el Alert, o manejarlo solo con message
            // setError(`Error al entrenar: ${err.message || 'Error desconocido'}`);

        }
        // onSettledCallback se puede usar para limpiar estados si fuera necesario
    });

    const forecastMutation = useGenerateForecastMutation({
        onSuccessCallback: (forecastResult) => {
            setHistoricalData({
                dates: forecastResult.historical_dates || [],
                values: forecastResult.historical_values || []
            });
            setForecastData(forecastResult.predictions || []);
            // Si el endpoint de forecast devuelve métricas actualizadas, úsalas
            // setMetrics(forecastResult.metrics || metrics); // Ojo: ¿Sobrescribir métricas de entrenamiento? Decide tu lógica
            setLastModelUsed(forecastResult.model_used || `Predicted with ${config.selectedModelType.toUpperCase()}`);
        },
        onErrorCallback: (err) => {
             // setError(`Error al pronosticar: ${err.message || 'Error desconocido'}`);
             setHistoricalData({ dates: [], values: [] }); // Limpiar datos en error
             setForecastData([]);
         }
    });

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
            return updatedConfig;
        });

        // Resetear datos si el ticker cambia
        if (newConfig.selectedTicker !== undefined && newConfig.selectedTicker !== config.selectedTicker) {
             setHistoricalData({ dates: [], values: [] });
             setForecastData([]);
             //setMetrics({});
             setLastModelUsed('');
             //setError(null);
        }
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

    // El useCallback asegura que la función no se vuelva a crear en cada renderizado, lo que mejora el rendimiento
    const handleTrain = useCallback(async (modelType, currentConfig) => {
        if (dateRangeWarning && dateRangeWarning.includes('podría ser muy corto')) {
            if (!window.confirm(dateRangeWarning + "\n\nEl entrenamiento podría fallar. ¿Deseas continuar de todas formas?")) {
                return; // El usuario canceló
            }
         }
          if (dateRangeWarning && dateRangeWarning.includes('posterior a la fecha de inicio')) {
             message.error('Corrige las fechas antes de entrenar.');
             return; // No continuar si las fechas son inválidas
          }

          trainMutation.mutate({ modelType, config: currentConfig }); // Llama a la mutación de entrenamiento
      }, [trainMutation, dateRangeWarning]); // Dependencias para el useCallback

    // Maneja el pronóstico, similar a handleTrain pero para generar pronósticos
    const handleForecast = useCallback(async (modelType, currentConfig) => {
        const key = 'forecastMessage';

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

    // --- Define los items para las pestañas ---
    const resultTabs = [
        {
            key: '1', // Identificador único para la pestaña
            label: 'Gráfico y Pronóstico', // Texto que se muestra en la pestaña
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
             // Deshabilitar si no hay métricas? Opcional:
             disabled: !latestRunId || Object.keys(currentMetricsToDisplay).length === 0
        },
        {
            key: '3',
            label: 'Detalles del Modelo',
            // Usa el nuevo componente y pásale los datos necesarios
            children: <ModelDetailsDisplay latestRun={latestRun} />,
            // La condición disabled ahora solo depende de si existe una ejecución reciente
            disabled: !latestRunId
        },
        {
            key: '4',
            label: 'Comparación de Modelos',
            children: <ModelComparisonTable results={trainingResults} />,
            disabled: Object.keys(trainingResults).length === 0 // Deshabilitar si no hay resultados
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
                    style={{ position: 'absolute', right: '20px', top: '16px' }} // Posicionar el switch
                />
                </Header>
                <Content style={{ padding: '20px 50px' }}>
                    {/* Mostrar error del hook si existe */}
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
                    <Row gutter={[16, 16]}>
                        <Col xs={24} md={10} lg={8} xl={7}>
                            {/* Usar el isLoading del hook correspondiente */}
                            <Spin spinning={trainMutation.isPending || forecastMutation.isPending} tip="Procesando...">
                                <Card title="Configuración">
                                    <ConfigurationPanel // La versión AntD
                                        availableModelTypes={availableModelTypes}
                                        availableTickers={availableTickers}
                                        onConfigChange={handleConfigChange}
                                        onTrain={handleTrain} // Pasas la función handleTrain de App
                                        onForecast={handleForecast} // Pasas la función handleForecast de App
                                        initialConfig={config}
                                        // ---- PASAR LA ADVERTENCIA COMO PROP ----
                                        dateRangeWarning={dateRangeWarning}
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
                </Content>
                <Footer style={{ textAlign: 'center' }}>
                    Series de Tiempo Financieras V.0.1 ©{new Date().getFullYear()} Creado con Ant Design
                </Footer>
            </Layout>
        </ConfigProvider>
    );
}

export default App;
