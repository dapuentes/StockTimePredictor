import React, { useState, useCallback } from 'react';
import { 
    Card, 
    Select, 
    Button, 
    Spin, 
    Alert, 
    Space, 
    Typography, 
    Tabs,
    Tooltip,
    Row,
    Col,
    Statistic,
    Tag,
    Segmented
} from 'antd';
import { 
    ExperimentOutlined, 
    BarChartOutlined, 
    DotChartOutlined,
    InfoCircleOutlined,
    ReloadOutlined,
    QuestionCircleOutlined
} from '@ant-design/icons';
import { getShapExplanation, getGlobalImportance, getShapPlot, getWaterfallPlot } from '../services/api';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

/**
 * ShapExplainer Component
 * Provides SHAP-based model interpretability visualizations
 */
const ShapExplainer = ({ ticker = 'NU', onError }) => {
    // State
    const [selectedModel, setSelectedModel] = useState('xgboost');
    const [plotType, setPlotType] = useState('bar');
    const [loading, setLoading] = useState({
        explanation: false,
        globalImportance: false,
        summaryPlot: false,
        waterfallPlot: false
    });
    const [data, setData] = useState({
        explanation: null,
        globalImportance: null,
        summaryPlot: null,
        waterfallPlot: null
    });
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('importance');

    // Available models for SHAP (tree-based only)
    const availableModels = [
        { value: 'xgboost', label: 'XGBoost', description: 'Gradient Boosting' },
        { value: 'rf', label: 'Random Forest', description: 'Ensemble de árboles' }
    ];

    // Fetch SHAP explanation
    const fetchExplanation = useCallback(async () => {
        setLoading(prev => ({ ...prev, explanation: true }));
        setError(null);
        try {
            const result = await getShapExplanation(ticker, selectedModel, 10);
            setData(prev => ({ ...prev, explanation: result }));
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, explanation: false }));
        }
    }, [ticker, selectedModel, onError]);

    // Fetch global importance
    const fetchGlobalImportance = useCallback(async () => {
        setLoading(prev => ({ ...prev, globalImportance: true }));
        setError(null);
        try {
            const result = await getGlobalImportance(selectedModel, ticker);
            setData(prev => ({ ...prev, globalImportance: result }));
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, globalImportance: false }));
        }
    }, [ticker, selectedModel, onError]);

    // Fetch summary plot
    const fetchSummaryPlot = useCallback(async () => {
        setLoading(prev => ({ ...prev, summaryPlot: true }));
        setError(null);
        try {
            const result = await getShapPlot(selectedModel, ticker, plotType, 15);
            setData(prev => ({ ...prev, summaryPlot: result }));
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, summaryPlot: false }));
        }
    }, [ticker, selectedModel, plotType, onError]);

    // Fetch waterfall plot
    const fetchWaterfallPlot = useCallback(async () => {
        setLoading(prev => ({ ...prev, waterfallPlot: true }));
        setError(null);
        try {
            const result = await getWaterfallPlot(selectedModel, ticker, 0, 10);
            setData(prev => ({ ...prev, waterfallPlot: result }));
        } catch (err) {
            setError(err.message);
            onError?.(err);
        } finally {
            setLoading(prev => ({ ...prev, waterfallPlot: false }));
        }
    }, [ticker, selectedModel, onError]);

    // Fetch all data
    const fetchAllData = useCallback(async () => {
        await Promise.all([
            fetchGlobalImportance(),
            fetchSummaryPlot(),
            fetchWaterfallPlot()
        ]);
    }, [fetchGlobalImportance, fetchSummaryPlot, fetchWaterfallPlot]);

    // Render feature importance bars
    const renderFeatureImportance = () => {
        if (!data.globalImportance?.feature_importance) return null;

        const features = data.globalImportance.feature_importance;
        const maxValue = Math.max(...features.map(f => f.importance));

        return (
            <div className="shap-feature-list">
                {features.slice(0, 10).map((feature, index) => (
                    <div key={feature.feature} className="shap-feature-item">
                        <div className="feature-header">
                            <span className="feature-rank">#{index + 1}</span>
                            <Text strong className="feature-name">{feature.feature}</Text>
                            <Text type="secondary" className="feature-value">
                                {(feature.importance * 100).toFixed(2)}%
                            </Text>
                        </div>
                        <div className="feature-bar-container">
                            <div 
                                className="feature-bar" 
                                style={{ 
                                    width: `${(feature.importance / maxValue) * 100}%`,
                                    background: `linear-gradient(90deg, #1890ff ${feature.importance * 100}%, #52c41a)`
                                }}
                            />
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    // Tab items
    const tabItems = [
        {
            key: 'importance',
            label: (
                <span>
                    <BarChartOutlined /> Importancia Global
                </span>
            ),
            children: (
                <div className="tab-content">
                    <Space direction="vertical" style={{ width: '100%' }}>
                        <div className="section-header">
                            <Title level={5}>
                                <ExperimentOutlined /> Importancia de Features (SHAP)
                            </Title>
                            <Tooltip title="Los valores SHAP miden la contribución de cada feature a la predicción">
                                <QuestionCircleOutlined style={{ color: '#8c8c8c' }} />
                            </Tooltip>
                        </div>
                        
                        <Button 
                            type="primary" 
                            icon={<ReloadOutlined />}
                            onClick={fetchGlobalImportance}
                            loading={loading.globalImportance}
                        >
                            Calcular Importancia
                        </Button>

                        {loading.globalImportance ? (
                            <div className="loading-container">
                                <Spin tip="Calculando valores SHAP..." />
                            </div>
                        ) : data.globalImportance ? (
                            <>
                                <Row gutter={16} style={{ marginTop: 16 }}>
                                    <Col span={8}>
                                        <Statistic 
                                            title="Modelo" 
                                            value={data.globalImportance.model_type?.toUpperCase()} 
                                        />
                                    </Col>
                                    <Col span={8}>
                                        <Statistic 
                                            title="Features Analizados" 
                                            value={data.globalImportance.feature_importance?.length || 0} 
                                        />
                                    </Col>
                                    <Col span={8}>
                                        <Statistic 
                                            title="Muestras" 
                                            value={data.globalImportance.n_samples || 'N/A'} 
                                        />
                                    </Col>
                                </Row>
                                {renderFeatureImportance()}
                            </>
                        ) : (
                            <Alert
                                type="info"
                                message="Haz clic en 'Calcular Importancia' para ver qué features son más importantes para el modelo."
                                showIcon
                            />
                        )}
                    </Space>
                </div>
            )
        },
        {
            key: 'summary',
            label: (
                <span>
                    <DotChartOutlined /> Gráfico Summary
                </span>
            ),
            children: (
                <div className="tab-content">
                    <Space direction="vertical" style={{ width: '100%' }}>
                        <div className="section-header">
                            <Title level={5}>Summary Plot SHAP</Title>
                            <Segmented
                                value={plotType}
                                onChange={setPlotType}
                                options={[
                                    { value: 'bar', label: 'Barras' },
                                    { value: 'dot', label: 'Puntos' }
                                ]}
                            />
                        </div>

                        <Button 
                            type="primary" 
                            icon={<ReloadOutlined />}
                            onClick={fetchSummaryPlot}
                            loading={loading.summaryPlot}
                        >
                            Generar Gráfico
                        </Button>

                        {loading.summaryPlot ? (
                            <div className="loading-container">
                                <Spin tip="Generando gráfico SHAP..." />
                            </div>
                        ) : data.summaryPlot?.plot_base64 ? (
                            <div className="plot-container">
                                <img 
                                    src={`data:image/png;base64,${data.summaryPlot.plot_base64}`}
                                    alt="SHAP Summary Plot"
                                    style={{ maxWidth: '100%', borderRadius: 8 }}
                                />
                                <Paragraph type="secondary" style={{ marginTop: 8 }}>
                                    {plotType === 'dot' 
                                        ? 'Cada punto representa una muestra. El color indica el valor del feature (rojo=alto, azul=bajo).'
                                        : 'Las barras muestran la importancia media absoluta de cada feature.'}
                                </Paragraph>
                            </div>
                        ) : (
                            <Alert
                                type="info"
                                message="Genera el gráfico para visualizar la distribución de valores SHAP."
                                showIcon
                            />
                        )}
                    </Space>
                </div>
            )
        },
        {
            key: 'waterfall',
            label: (
                <span>
                    <BarChartOutlined /> Waterfall
                </span>
            ),
            children: (
                <div className="tab-content">
                    <Space direction="vertical" style={{ width: '100%' }}>
                        <div className="section-header">
                            <Title level={5}>Waterfall Plot - Explicación Individual</Title>
                            <Tooltip title="Muestra cómo cada feature contribuye a una predicción específica">
                                <QuestionCircleOutlined style={{ color: '#8c8c8c' }} />
                            </Tooltip>
                        </div>

                        <Button 
                            type="primary" 
                            icon={<ReloadOutlined />}
                            onClick={fetchWaterfallPlot}
                            loading={loading.waterfallPlot}
                        >
                            Generar Waterfall
                        </Button>

                        {loading.waterfallPlot ? (
                            <div className="loading-container">
                                <Spin tip="Generando waterfall plot..." />
                            </div>
                        ) : data.waterfallPlot?.plot_base64 ? (
                            <div className="plot-container">
                                <img 
                                    src={`data:image/png;base64,${data.waterfallPlot.plot_base64}`}
                                    alt="SHAP Waterfall Plot"
                                    style={{ maxWidth: '100%', borderRadius: 8 }}
                                />
                                <Paragraph type="secondary" style={{ marginTop: 8 }}>
                                    El gráfico waterfall muestra cómo cada feature empuja la predicción 
                                    desde el valor base hacia el valor final predicho.
                                </Paragraph>
                            </div>
                        ) : (
                            <Alert
                                type="info"
                                message="Genera el waterfall plot para ver la explicación de una predicción individual."
                                showIcon
                            />
                        )}
                    </Space>
                </div>
            )
        }
    ];

    return (
        <Card 
            className="shap-explainer-card"
            title={
                <Space>
                    <ExperimentOutlined style={{ color: '#1890ff' }} />
                    <span>Interpretabilidad SHAP</span>
                    <Tag color="blue">{ticker}</Tag>
                </Space>
            }
            extra={
                <Space>
                    <Select
                        value={selectedModel}
                        onChange={setSelectedModel}
                        style={{ width: 150 }}
                    >
                        {availableModels.map(model => (
                            <Option key={model.value} value={model.value}>
                                {model.label}
                            </Option>
                        ))}
                    </Select>
                    <Button 
                        icon={<ReloadOutlined />}
                        onClick={fetchAllData}
                        loading={Object.values(loading).some(l => l)}
                    >
                        Actualizar Todo
                    </Button>
                </Space>
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

            <div className="shap-info-banner" style={{ marginBottom: 16 }}>
                <InfoCircleOutlined style={{ marginRight: 8, color: '#1890ff' }} />
                <Text type="secondary">
                    SHAP (SHapley Additive exPlanations) proporciona una interpretación basada en 
                    teoría de juegos para explicar las predicciones de modelos de machine learning.
                    Solo disponible para modelos basados en árboles (XGBoost, Random Forest).
                </Text>
            </div>

            <Tabs 
                activeKey={activeTab} 
                onChange={setActiveTab}
                items={tabItems}
            />

            <style>{`
                .shap-explainer-card {
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                
                .shap-feature-list {
                    margin-top: 16px;
                }
                
                .shap-feature-item {
                    margin-bottom: 12px;
                    padding: 8px 12px;
                    background: #fafafa;
                    border-radius: 8px;
                }
                
                .feature-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 6px;
                }
                
                .feature-rank {
                    background: #1890ff;
                    color: white;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 600;
                }
                
                .feature-name {
                    flex: 1;
                }
                
                .feature-value {
                    font-family: monospace;
                }
                
                .feature-bar-container {
                    height: 8px;
                    background: #f0f0f0;
                    border-radius: 4px;
                    overflow: hidden;
                }
                
                .feature-bar {
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.5s ease;
                }
                
                .loading-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 40px;
                }
                
                .plot-container {
                    margin-top: 16px;
                    text-align: center;
                }
                
                .section-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 16px;
                }
                
                .shap-info-banner {
                    background: #e6f7ff;
                    padding: 12px 16px;
                    border-radius: 8px;
                    border-left: 4px solid #1890ff;
                }
                
                .tab-content {
                    padding: 16px 0;
                }
            `}</style>
        </Card>
    );
};

export default ShapExplainer;
