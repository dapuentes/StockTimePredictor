/**
 * StatsOverview Component - Tarjetas de estadísticas rápidas
 * Muestra información resumida del estado actual
 */
import React from 'react';
import { Card, Row, Col, Statistic, Tag, Tooltip, Progress } from 'antd';
import {
  StockOutlined,
  RocketOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

function StatsOverview({ 
  selectedTicker,
  selectedModelType,
  activeJobsCount,
  completedModelsCount,
  lastPrediction,
  forecastHorizon
}) {
  const stats = [
    {
      key: 'ticker',
      title: 'Ticker Activo',
      value: selectedTicker || 'N/A',
      icon: <StockOutlined />,
      color: '#1a365d',
      description: 'Acción seleccionada'
    },
    {
      key: 'model',
      title: 'Modelo',
      value: selectedModelType?.toUpperCase() || 'N/A',
      icon: <ThunderboltOutlined />,
      color: '#2c5282',
      description: 'Tipo de modelo ML'
    },
    {
      key: 'active',
      title: 'Entrenamientos',
      value: activeJobsCount || 0,
      icon: <RocketOutlined spin={activeJobsCount > 0} />,
      color: activeJobsCount > 0 ? '#38a169' : '#718096',
      description: 'En progreso',
      suffix: activeJobsCount > 0 ? (
        <Tag color="processing" style={{ marginLeft: '8px' }}>En curso</Tag>
      ) : null
    },
    {
      key: 'forecast',
      title: 'Horizonte',
      value: forecastHorizon || 10,
      icon: <LineChartOutlined />,
      color: '#4299e1',
      description: 'Días de pronóstico',
      suffix: <span style={{ fontSize: '14px', color: '#718096' }}> días</span>
    }
  ];

  return (
    <div className="stats-overview animate-fadeIn" style={{ marginBottom: '24px' }}>
      <Row gutter={[16, 16]}>
        {stats.map(stat => (
          <Col xs={12} sm={12} md={6} key={stat.key}>
            <Card
              hoverable
              style={{ 
                borderRadius: '12px',
                border: '1px solid #e2e8f0',
                transition: 'all 0.25s ease'
              }}
              bodyStyle={{ padding: '16px' }}
            >
              <Tooltip title={stat.description}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ 
                      fontSize: '12px', 
                      color: '#718096', 
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                      marginBottom: '4px'
                    }}>
                      {stat.title}
                    </div>
                    <div style={{ 
                      fontSize: '24px', 
                      fontWeight: '700',
                      color: stat.color,
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      {stat.value}
                      {stat.suffix}
                    </div>
                  </div>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '10px',
                    backgroundColor: `${stat.color}15`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: stat.color,
                    fontSize: '18px'
                  }}>
                    {stat.icon}
                  </div>
                </div>
              </Tooltip>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
}

export default StatsOverview;
