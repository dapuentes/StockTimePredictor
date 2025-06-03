import React from 'react';
import { Progress, Card, Typography, Space, Statistic, Row, Col } from 'antd';
import { ClockCircleOutlined, RocketOutlined, BarChartOutlined } from '@ant-design/icons';

const { Text, Title } = Typography;

const DetailedProgressBar = ({ 
    trainingJob, 
    trainingStatusMessage, 
    isVisible = true 
}) => {
    if (!isVisible || !trainingJob) {
        return null;
    }

    const {
        modelType,
        config,
        progress = 0,
        currentStep,
        trainingStep,
        totalTrainingSteps,
        subProgress,
        startTime
    } = trainingJob;

    // Calcular tiempo transcurrido
    const getElapsedTime = () => {
        if (!startTime) return 'N/A';
        const now = new Date();
        const elapsed = Math.floor((now - new Date(startTime)) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    // Determinar el color del progreso principal
    const getProgressColor = (progress) => {
        if (progress < 25) return '#ff4d4f'; // Rojo
        if (progress < 50) return '#fa8c16'; // Naranja
        if (progress < 75) return '#fadb14'; // Amarillo
        return '#52c41a'; // Verde
    };

    // Determinar el estado actual
    const getCurrentPhase = (progress) => {
        if (progress < 20) return 'Preparación';
        if (progress < 90) return 'Entrenamiento';
        return 'Finalización';
    };

    return (
        <Card 
            size="small" 
            style={{ 
                marginBottom: 16, 
                border: '1px solid #d9d9d9',
                borderRadius: 8,
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}
        >
            <Space direction="vertical" style={{ width: '100%' }} size="small">
                {/* Header con información del modelo */}
                <Row justify="space-between" align="middle">
                    <Col>
                        <Title level={5} style={{ margin: 0, color: '#1890ff' }}>
                            <RocketOutlined /> Entrenando {modelType?.toUpperCase()} - {config?.selectedTicker}
                        </Title>
                    </Col>
                    <Col>
                        <Statistic
                            title="Tiempo"
                            value={getElapsedTime()}
                            prefix={<ClockCircleOutlined />}
                            valueStyle={{ fontSize: 14 }}
                        />
                    </Col>
                </Row>

                {/* Progreso principal */}
                <Space direction="vertical" style={{ width: '100%' }} size={4}>
                    <Row justify="space-between">
                        <Text strong>Progreso General</Text>
                        <Text type="secondary">{getCurrentPhase(progress)}</Text>
                    </Row>
                    
                    <Progress
                        percent={Math.round(progress)}
                        strokeColor={getProgressColor(progress)}
                        strokeWidth={8}
                        format={(percent) => `${percent}%`}
                    />
                </Space>

                {/* Sub-progreso si está disponible */}
                {subProgress !== undefined && subProgress !== null && (
                    <Space direction="vertical" style={{ width: '100%' }} size={4}>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                            Progreso del Paso Actual
                        </Text>
                        <Progress
                            percent={Math.round(subProgress)}
                            size="small"
                            strokeColor="#722ed1"
                            format={(percent) => `${percent}%`}
                        />
                    </Space>
                )}

                {/* Información de pasos */}
                {trainingStep && totalTrainingSteps && (
                    <Row gutter={16}>
                        <Col span={12}>
                            <Statistic
                                title="Paso Actual"
                                value={trainingStep}
                                suffix={`/ ${totalTrainingSteps}`}
                                valueStyle={{ fontSize: 14, color: '#1890ff' }}
                            />
                        </Col>
                        <Col span={12}>
                            <Statistic
                                title="Progreso de Pasos"
                                value={Math.round((trainingStep / totalTrainingSteps) * 100)}
                                suffix="%"
                                prefix={<BarChartOutlined />}
                                valueStyle={{ fontSize: 14, color: '#52c41a' }}
                            />
                        </Col>
                    </Row>
                )}

                {/* Mensaje de estado actual */}
                {(currentStep || trainingStatusMessage) && (
                    <Card size="small" style={{ backgroundColor: '#fafafa', border: 'none' }}>
                        <Text 
                            style={{ 
                                fontSize: 12, 
                                fontFamily: 'monospace',
                                color: '#666'
                            }}
                        >
                            {currentStep || trainingStatusMessage}
                        </Text>
                    </Card>
                )}

                {/* Indicadores de estado */}
                <Row gutter={8}>
                    <Col>
                        <Text 
                            style={{ 
                                fontSize: 11, 
                                color: progress > 0 ? '#52c41a' : '#d9d9d9' 
                            }}
                        >
                            ● Iniciado
                        </Text>
                    </Col>
                    <Col>
                        <Text 
                            style={{ 
                                fontSize: 11, 
                                color: progress >= 20 ? '#52c41a' : '#d9d9d9' 
                            }}
                        >
                            ● Entrenamiento
                        </Text>
                    </Col>
                    <Col>
                        <Text 
                            style={{ 
                                fontSize: 11, 
                                color: progress >= 90 ? '#52c41a' : '#d9d9d9' 
                            }}
                        >
                            ● Finalizando
                        </Text>
                    </Col>
                </Row>
            </Space>
        </Card>
    );
};

export default DetailedProgressBar;
