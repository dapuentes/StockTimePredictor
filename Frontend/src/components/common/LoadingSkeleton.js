/**
 * LoadingSkeleton Component - Esqueletos de carga elegantes
 * Mejora la percepción del tiempo de carga
 */
import React from 'react';
import { Skeleton, Card, Row, Col, Space } from 'antd';

// Skeleton para las tarjetas de estadísticas
export function StatsSkeleton() {
  return (
    <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
      {[1, 2, 3, 4].map(i => (
        <Col xs={12} sm={12} md={6} key={i}>
          <Card bodyStyle={{ padding: '16px' }}>
            <Skeleton.Input active size="small" style={{ width: '60px', marginBottom: '8px' }} />
            <Skeleton.Input active size="large" style={{ width: '100px' }} />
          </Card>
        </Col>
      ))}
    </Row>
  );
}

// Skeleton para el panel de configuración
export function ConfigPanelSkeleton() {
  return (
    <Card>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <div>
          <Skeleton.Input active size="small" style={{ width: '100px', marginBottom: '8px' }} />
          <Skeleton.Input active style={{ width: '100%' }} />
        </div>
        <div>
          <Skeleton.Input active size="small" style={{ width: '100px', marginBottom: '8px' }} />
          <Skeleton.Input active style={{ width: '100%' }} />
        </div>
        <div>
          <Skeleton.Input active size="small" style={{ width: '100px', marginBottom: '8px' }} />
          <Skeleton.Input active style={{ width: '100%' }} />
        </div>
        <Skeleton.Button active block size="large" />
        <Skeleton.Button active block />
      </Space>
    </Card>
  );
}

// Skeleton para el gráfico
export function ChartSkeleton() {
  return (
    <Card>
      <Skeleton.Input active size="small" style={{ width: '200px', marginBottom: '16px' }} />
      <div 
        style={{ 
          height: '400px', 
          background: 'linear-gradient(145deg, #f7fafc 0%, #edf2f7 100%)',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Skeleton.Avatar active shape="square" style={{ width: '80%', height: '80%' }} />
      </div>
    </Card>
  );
}

// Skeleton para la lista de modelos
export function ModelsListSkeleton() {
  return (
    <div style={{ padding: '12px' }}>
      <Skeleton.Input active size="small" style={{ width: '150px', marginBottom: '12px' }} />
      {[1, 2, 3].map(i => (
        <div 
          key={i}
          style={{ 
            padding: '8px 12px', 
            background: '#f7fafc', 
            borderRadius: '6px', 
            marginBottom: '8px' 
          }}
        >
          <Skeleton.Input active size="small" style={{ width: '100%' }} />
        </div>
      ))}
    </div>
  );
}

// Skeleton para métricas
export function MetricsSkeleton() {
  return (
    <Row gutter={[16, 16]}>
      {[1, 2, 3, 4].map(i => (
        <Col xs={12} md={6} key={i}>
          <Card bodyStyle={{ padding: '16px', textAlign: 'center' }}>
            <Skeleton.Input active size="small" style={{ width: '60px', marginBottom: '8px' }} />
            <br />
            <Skeleton.Input active size="large" style={{ width: '80px' }} />
          </Card>
        </Col>
      ))}
    </Row>
  );
}

// Componente principal que combina todos los skeletons
function LoadingSkeleton({ type = 'full' }) {
  switch (type) {
    case 'stats':
      return <StatsSkeleton />;
    case 'config':
      return <ConfigPanelSkeleton />;
    case 'chart':
      return <ChartSkeleton />;
    case 'models':
      return <ModelsListSkeleton />;
    case 'metrics':
      return <MetricsSkeleton />;
    case 'full':
    default:
      return (
        <>
          <StatsSkeleton />
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={8}>
              <ConfigPanelSkeleton />
            </Col>
            <Col xs={24} lg={16}>
              <ChartSkeleton />
            </Col>
          </Row>
        </>
      );
  }
}

export default LoadingSkeleton;
