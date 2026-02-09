import React, { useState, useCallback, useEffect } from 'react';
import { 
    Card, 
    Select, 
    Button, 
    Spin, 
    Alert, 
    Space, 
    Typography, 
    Row,
    Col,
    Statistic,
    Tag,
    Checkbox,
    InputNumber,
    Divider,
    Table,
    Progress
} from 'antd';
import { 
    ClusterOutlined, 
    LineChartOutlined, 
    ThunderboltOutlined,
    SyncOutlined,
    InfoCircleOutlined,
    CheckCircleOutlined,
    ExclamationCircleOutlined,
    TrophyOutlined
} from '@ant-design/icons';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title as ChartTitle,
    Tooltip as ChartTooltip,
    Legend,
    Filler
} from 'chart.js';
import { getEnsemblePrediction, compareModelPredictions, getEnsembleModels } from '../services/api';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    ChartTitle,
    ChartTooltip,
    Legend,
    Filler
);

const { Title, Text } = Typography;
const { Option } = Select;

/**
 * EnsemblePredictor Component
 * Combines predictions from multiple models using various ensemble methods
 */
const EnsemblePredictor = ({ ticker = 'NU', onPrediction, onError }) => {
    // State
    const [ensembleMethod, setEnsembleMethod] = useState('weighted_average');
    const [selectedModels, setSelectedModels] = useState(['rf', 'xgboost', 'lstm', 'prophet']);
    const [forecastHorizon, setForecastHorizon] = useState(10);
    const [loading, setLoading] = useState({
        prediction: false,
        comparison: false,
        models: false
    });
    const [data, setData] = useState({
        prediction: null,
        comparison: null,
        availableModels: null
    });
    const [error, setError] = useState(null);

    // Ensemble methods available
    const ensembleMethods = [
        { 
            value: 'simple_average', 
            label: 'Promedio Simple', 
            description: 'Promedio aritmético de todas las predicciones',
            icon: <ClusterOutlined />
        },
        { 
            value: 'weighted_average', 
            label: 'Promedio Ponderado', 
            description: 'Ponderación basada en MAE histórico (mejor peso = menor error)',
            icon: <ThunderboltOutlined />
        },
        { 
            value: 'median', 
            label: 'Mediana', 
            description: 'Mediana de las predicciones (robusto a outliers)',
            icon: <LineChartOutlined />
        },
        { 
            value: 'best_model', 
            label: 'Mejor Modelo', 
            description: 'Usa solo el modelo con menor MAE histórico',
            icon: <TrophyOutlined />
        }
    ];

    // Available models
    const allModels = [
        { value: 'rf', label: 'Random Forest', color: '#2f9e5a' },
        { value: 'xgboost', label: 'XGBoost', color: '#2b6cb0' },
        { value: 'lstm', label: 'LSTM', color: '#6b46c1' },
        { value: 'prophet', label: 'Prophet', color: '#d69e2e' }
    ];

    // Fetch available models on mount
    useEffect(() => {
        const fetchModels = async () => {
            setLoading(prev => ({ ...prev, models: true }));
            try {
                const result = await getEnsembleModels();
                setData(prev => ({ ...prev, availableModels: result }));
            } catch (err) {
                console.error('Error fetching ensemble models:', err);
            } finally {
                setLoading(prev => ({ ...prev, models: false }));
            }
        };
        fetchModels();
    }, []);

    // Fetch ensemble prediction
    const fetchPrediction = useCallback(async () => {
        if (selectedModels.length < 2 && ensembleMethod !== 'best_model') {
            setError('Selecciona al menos 2 modelos para el ensemble');
            return;
        }

        setLoading(prev => ({ ...prev, prediction: true }));
        setError(null);
        
        try {
            const result = await getEnsemblePrediction({
                ticker,
                forecastHorizon,
                targetCol: 'Close',
                models: selectedModels,
                ensembleMethod
            });
            setData(prev => ({ ...prev, prediction: result }));
            onPrediction?.(result);
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, prediction: false }));
        }
    }, [ticker, forecastHorizon, selectedModels, ensembleMethod, onPrediction, onError]);

    // Fetch comparison
    const fetchComparison = useCallback(async () => {
        setLoading(prev => ({ ...prev, comparison: true }));
        setError(null);
        
        try {
            const result = await compareModelPredictions(ticker, forecastHorizon);
            setData(prev => ({ ...prev, comparison: result }));
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, comparison: false }));
        }
    }, [ticker, forecastHorizon, onError]);

    // Chart data for ensemble prediction
    const getChartData = () => {
        if (!data.prediction) return null;

        const pred = data.prediction;
        const datasets = [];

        // Ensemble prediction
        if (pred.ensemble_predictions) {
            datasets.push({
                label: `Ensemble (${ensembleMethod.replace('_', ' ')})`,
                data: pred.ensemble_predictions,
                borderColor: '#c53030',
                backgroundColor: 'rgba(197, 48, 48, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4
            });
        }

        // Individual model predictions
        if (pred.individual_predictions) {
            Object.entries(pred.individual_predictions).forEach(([model, predictions]) => {
                const modelConfig = allModels.find(m => m.value === model);
                datasets.push({
                    label: modelConfig?.label || model,
                    data: predictions,
                    borderColor: modelConfig?.color || '#8492a6',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                });
            });
        }

        // Confidence interval
        if (pred.confidence_interval) {
            datasets.push({
                label: 'Intervalo de Confianza',
                data: pred.confidence_interval.upper,
                borderColor: 'rgba(197, 48, 48, 0.3)',
                backgroundColor: 'rgba(197, 48, 48, 0.08)',
                borderWidth: 1,
                fill: '+1',
                tension: 0.4,
                pointRadius: 0
            });
            datasets.push({
                label: 'IC Lower',
                data: pred.confidence_interval.lower,
                borderColor: 'rgba(197, 48, 48, 0.3)',
                backgroundColor: 'transparent',
                borderWidth: 1,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            });
        }

        return {
            labels: pred.forecast_dates || Array.from({ length: forecastHorizon }, (_, i) => `Día ${i + 1}`),
            datasets
        };
    };

    // Chart options
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    pointStyle: 'circle',
                    padding: 20,
                    font: { family: 'Inter, sans-serif', size: 12 }
                }
            },
            title: {
                display: true,
                text: `Predicción Ensemble — ${ticker}`,
                font: { family: 'Inter, sans-serif', size: 15, weight: '600' },
                padding: { bottom: 16 },
                color: '#232e3e'
            },
            tooltip: {
                mode: 'index',
                intersect: false,
            }
        },
        scales: {
            y: {
                title: { display: true, text: 'Precio ($)' },
                grid: { color: 'rgba(0,0,0,0.06)' },
                ticks: { font: { size: 11 } }
            },
            x: {
                title: { display: true, text: 'Fecha' },
                grid: { color: 'rgba(0,0,0,0.04)' },
                ticks: { font: { size: 11 }, maxTicksLimit: 12 }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };

    // Comparison table columns
    const comparisonColumns = [
        {
            title: 'Modelo',
            dataIndex: 'model',
            key: 'model',
            render: (text) => {
                const modelConfig = allModels.find(m => m.value === text);
                return (
                    <Tag color={modelConfig?.color}>
                        {modelConfig?.label || text}
                    </Tag>
                );
            }
        },
        {
            title: 'Predicción Final',
            dataIndex: 'final_prediction',
            key: 'final_prediction',
            render: (val) => val ? `$${val.toFixed(2)}` : 'N/A'
        },
        {
            title: 'Variación %',
            dataIndex: 'change_percent',
            key: 'change_percent',
            render: (val) => {
                if (val === undefined) return 'N/A';
                const color = val >= 0 ? 'var(--color-success-500)' : 'var(--color-danger-500)';
                return <span style={{ color }}>{val >= 0 ? '+' : ''}{val.toFixed(2)}%</span>;
            }
        },
        {
            title: 'MAE Histórico',
            dataIndex: 'historical_mae',
            key: 'historical_mae',
            render: (val) => val ? val.toFixed(4) : 'N/A'
        },
        {
            title: 'Estado',
            dataIndex: 'status',
            key: 'status',
            render: (status) => (
                status === 'success' 
                    ? <CheckCircleOutlined style={{ color: 'var(--color-success-500)' }} />
                    : <ExclamationCircleOutlined style={{ color: 'var(--color-warning-500)' }} />
            )
        }
    ];

    return (
        <Card 
            className="ensemble-predictor-card"
            title={
                <Space>
                    <ClusterOutlined style={{ color: 'var(--color-primary-600)' }} />
                    <span>Predictor Ensemble</span>
                    <Tag color="purple">{ticker}</Tag>
                </Space>
            }
            extra={
                <Button 
                    type="primary"
                    icon={<ThunderboltOutlined />}
                    onClick={fetchPrediction}
                    loading={loading.prediction}
                >
                    Generar Predicción
                </Button>
            }
        >
            {error && (
                <Alert
                    type="error"
                    message="Error"
                    description={error}
                    closable
                    onClose={() => setError(null)}
                    style={{ marginBottom: 16 }}
                />
            )}

            <div className="info-banner info-banner--ensemble">
                <InfoCircleOutlined style={{ color: '#6b46c1', flexShrink: 0 }} />
                <Text type="secondary">
                    El modelo ensemble combina las predicciones de múltiples modelos para obtener 
                    una predicción más robusta y reducir la varianza del error.
                </Text>
            </div>

            {/* Configuration Section */}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col span={8}>
                    <div className="config-section">
                        <Text strong>Método de Ensemble</Text>
                        <Select
                            value={ensembleMethod}
                            onChange={setEnsembleMethod}
                            style={{ width: '100%', marginTop: 8 }}
                        >
                            {ensembleMethods.map(method => (
                                <Option key={method.value} value={method.value}>
                                    <Space>
                                        {method.icon}
                                        <span>{method.label}</span>
                                    </Space>
                                </Option>
                            ))}
                        </Select>
                        <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                            {ensembleMethods.find(m => m.value === ensembleMethod)?.description}
                        </Text>
                    </div>
                </Col>
                
                <Col span={8}>
                    <div className="config-section">
                        <Text strong>Horizonte de Predicción</Text>
                        <InputNumber
                            min={1}
                            max={30}
                            value={forecastHorizon}
                            onChange={setForecastHorizon}
                            addonAfter="días"
                            style={{ width: '100%', marginTop: 8 }}
                        />
                    </div>
                </Col>
                
                <Col span={8}>
                    <div className="config-section">
                        <Text strong>Modelos a Incluir</Text>
                        <Checkbox.Group
                            value={selectedModels}
                            onChange={setSelectedModels}
                            style={{ marginTop: 8 }}
                        >
                            <Row>
                                {allModels.map(model => (
                                    <Col span={12} key={model.value}>
                                        <Checkbox value={model.value}>
                                            <Tag color={model.color} style={{ marginLeft: 4 }}>
                                                {model.label}
                                            </Tag>
                                        </Checkbox>
                                    </Col>
                                ))}
                            </Row>
                        </Checkbox.Group>
                    </div>
                </Col>
            </Row>

            <Divider />

            {/* Prediction Results */}
            {loading.prediction ? (
                <div className="loading-container">
                    <Spin tip="Generando predicción ensemble..." size="large" />
                </div>
            ) : data.prediction ? (
                <>
                    {/* Summary Stats */}
                    <Row gutter={16} style={{ marginBottom: 24 }}>
                        <Col span={6}>
                            <Statistic 
                                title="Método Usado" 
                                value={data.prediction.ensemble_method?.replace('_', ' ') || 'N/A'}
                                valueStyle={{ textTransform: 'capitalize' }}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic 
                                title="Modelos Combinados" 
                                value={data.prediction.models_used?.length || 0}
                                suffix={`/ ${allModels.length}`}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic 
                                title="Predicción Final" 
                                value={data.prediction.ensemble_predictions?.slice(-1)[0]?.toFixed(2) || 'N/A'}
                                prefix="$"
                                valueStyle={{ color: 'var(--color-success-600)' }}
                            />
                        </Col>
                        <Col span={6}>
                            <Statistic 
                                title="Confianza" 
                                value={data.prediction.confidence_score ? 
                                    (data.prediction.confidence_score * 100).toFixed(0) : 'N/A'}
                                suffix="%"
                            />
                        </Col>
                    </Row>

                    {/* Model Weights (for weighted average) */}
                    {data.prediction.model_weights && ensembleMethod === 'weighted_average' && (
                        <Card size="small" title="Pesos del Ensemble" style={{ marginBottom: 16 }}>
                            <Row gutter={16}>
                                {Object.entries(data.prediction.model_weights).map(([model, weight]) => {
                                    const modelConfig = allModels.find(m => m.value === model);
                                    return (
                                        <Col span={6} key={model}>
                                            <div style={{ textAlign: 'center' }}>
                                                <Tag color={modelConfig?.color}>{modelConfig?.label}</Tag>
                                                <Progress 
                                                    type="circle" 
                                                    percent={(weight * 100).toFixed(0)}
                                                    width={60}
                                                    format={percent => `${percent}%`}
                                                />
                                            </div>
                                        </Col>
                                    );
                                })}
                            </Row>
                        </Card>
                    )}

                    {/* Chart */}
                    <div style={{ height: 400, marginBottom: 24 }}>
                        {getChartData() && (
                            <Line data={getChartData()} options={chartOptions} />
                        )}
                    </div>
                </>
            ) : (
                <Alert
                    type="info"
                    message="Configura los parámetros y haz clic en 'Generar Predicción' para obtener el ensemble."
                    showIcon
                    style={{ marginBottom: 24 }}
                />
            )}

            <Divider />

            {/* Model Comparison Section */}
            <div className="comparison-section">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                    <Title level={5} style={{ margin: 0 }}>
                        <LineChartOutlined /> Comparación de Modelos
                    </Title>
                    <Button 
                        icon={<SyncOutlined />}
                        onClick={fetchComparison}
                        loading={loading.comparison}
                    >
                        Comparar Modelos
                    </Button>
                </div>

                {loading.comparison ? (
                    <div className="loading-container">
                        <Spin tip="Comparando modelos..." />
                    </div>
                ) : data.comparison?.comparison ? (
                    <Table
                        dataSource={data.comparison.comparison.map((item, idx) => ({ ...item, key: idx }))}
                        columns={comparisonColumns}
                        pagination={false}
                        size="small"
                    />
                ) : (
                    <Alert
                        type="info"
                        message="Haz clic en 'Comparar Modelos' para ver las predicciones individuales de cada modelo."
                        showIcon
                    />
                )}
            </div>
        </Card>
    );
};

export default EnsemblePredictor;
