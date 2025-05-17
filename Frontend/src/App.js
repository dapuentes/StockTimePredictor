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
    const [helpModalVisible, setHelpModalVisible] = useState(false); // Estado para el modal de ayuda

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

            setResidualsData({
            dates: result.residual_dates || [],
            values: result.residuals     || []
            });

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
            // setMetrics(forecastResult.metrics || metrics); // Ojo: ¿Sobrescribir métricas de entrenamiento? Decide tu lógica
            setLastModelUsed(forecastResult.model_used || `Predicted with ${config.selectedModelType.toUpperCase()}`);
        },
        onErrorCallback: (err) => {
             //setError(`Error al pronosticar: ${err.message || 'Error desconocido'}`);
             setHistoricalData({ dates: [], values: [] }); // Limpiar datos en error
             setForecastData([]);
         }
    });

    // --- Funciones de manejo de eventos ---
    const handleConfigChange = useCallback((newConfig) => {
        setConfig(prev => {
            const updatedConfig = { ...prev, ...newConfig };
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
            Modal.confirm({
                title: 'Advertencia sobre Rango de Fechas',
                content: dateRangeWarning + "\n\nEl entrenamiento podría fallar. ¿Deseas continuar de todas formas?",
                okText: 'Continuar',
                cancelText: 'Cancelar',
                onOk: () => {
                    // Solo si el usuario confirma, se ejecuta la mutación
                    trainMutation.mutate({ modelType, config: currentConfig });
                },
                onCancel: () => {
                    message.info('Entrenamiento cancelado por el usuario.');
                }
            });
        } else if (dateRangeWarning && dateRangeWarning.includes('posterior a la fecha de inicio')) {
             message.error('Corrige las fechas antes de entrenar.');
        } else {
            // Si no hay advertencias o errores de fecha, entrena directamente
             trainMutation.mutate({ modelType, config: currentConfig });
        }
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
    const featureImportances = latestRun?.featureImportances || [];
    const [residualsData, setResidualsData] = useState({ dates: [], values: [] });

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
            disabled: Object.keys(trainingResults).length === 0
        },
        {
            key: '5',
            label: 'Residuales',
            children: (
                <ResidualsDisplay
                dates={residualsData.dates}
                values={residualsData.values}
                />
            ),
            disabled: residualsData.values.length === 0
        }
    ];

    // --- Renderizado con Tabs ---
    return (
        <ConfigProvider
             theme={{
                 // Algoritmo para tema claro u oscuro
                 algorithm: currentTheme === 'dark' ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
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
                    <div style={{ padding: '0 50px' }}>
                    {/* Mostrar error del hook si existe */}
                    {(trainMutation.error || forecastMutation.error) && (
                        <Alert
                            message="Error"
                            description={trainMutation.error?.message || forecastMutation.error?.message || 'Ocurrió un error'}
                            type="error"
                            showIcon
                            closable
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
                                        dateRangeWarning={dateRangeWarning}
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