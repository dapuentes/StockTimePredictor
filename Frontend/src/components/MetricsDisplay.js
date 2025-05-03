import React from 'react';
import { Descriptions, Empty, Tooltip, Tag } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

// --- Textos de Interpretación ---
const interpretations = {
    MSE: "Error Cuadrático Medio: Es el promedio de los errores al cuadrado. Penaliza más los errores grandes que los pequeños. Sus unidades son el cuadrado de las unidades originales (ej. dólares²), por lo que es menos intuitivo.",
    RMSE: "Raíz del Error Cuadrático Medio: Es la raíz cuadrada del MSE. Vuelve a las unidades originales (ej. dólares) y representa el error típico, dando más peso a errores grandes. Menor es mejor.",
    MAE: "Error Absoluto Medio: Es el promedio del valor absoluto de los errores. Está en las unidades originales (ej. dólares) y es más fácil de interpretar que el RMSE como el 'error promedio'. Menor es mejor.",
    MAPE: "Error Porcentual Absoluto Medio: Es el promedio del error absoluto expresado como porcentaje del valor real. Es útil para comparar la precisión entre modelos o activos con diferentes escalas de precio. Menor es mejor (<10-15% suele considerarse bueno para acciones)."
};

// --- Helper para formatear números (opcional, pero útil) ---
const formatMetricValue = (value, isPercentage = false) => {
    if (value === null || typeof value === 'undefined' || isNaN(value)) {
        return 'N/A';
    }
    if (isPercentage) {
        // Añadir un pequeño épsilon puede ayudar con problemas de redondeo cerca de .005
        return (Number(value) * 100 + Number.EPSILON).toFixed(2) + '%';
    }
    return Number(value).toFixed(4);
};

// --- Helper para crear la etiqueta con Tooltip ---
const MetricLabel = ({ title, interpretation }) => (
    <span>
        {title}{' '}
        <Tooltip title={interpretation}>
            <QuestionCircleOutlined style={{ color: 'rgba(0,0,0,.45)', cursor: 'help' }} />
        </Tooltip>
    </span>
);


// --- Componente Principal ---
/**
 * Renders a display component for performance metrics with descriptions and formatted values.
 *
 * @param {Object} props - The properties object passed to the component.
 * @param {Object} props.metrics - An object containing the metrics to be displayed. Expected keys include MAPE, MAE, RMSE, and MSE, among others.
 * @return {React.Element} A component that visualizes the provided metrics, or a placeholder message if no metrics are available.
 */
function MetricsDisplay({ metrics }) { // Renombro prop a 'metrics' por claridad

    // Si no hay métricas, mostrar un placeholder
    if (!metrics || Object.keys(metrics).length === 0) {
        return <Empty description="Métricas no disponibles. Entrena un modelo o genera un pronóstico." />;
    }

    // Crear los items para el componente Descriptions
    const items = [
        {
            key: 'MAPE',
            // Usar el helper para el label con tooltip
            label: <MetricLabel title="MAPE" interpretation={interpretations.MAPE} />,
            // Mostrar el valor formateado como porcentaje
            // Añadir un Tag de color para evaluación rápida (opcional)
            children: (
                <Tag color={metrics.MAPE < 0.05 ? 'success' : metrics.MAPE < 0.10 ? 'processing' : metrics.MAPE < 0.15 ? 'warning' : 'error'}>
                    {formatMetricValue(metrics.MAPE, true)}
                </Tag>
            ),
            span: 2 // Ocupar más espacio si se desea
        },
        {
            key: 'MAE',
            label: <MetricLabel title="MAE ($)" interpretation={interpretations.MAE} />, // Indicar unidades ($)
            children: formatMetricValue(metrics.MAE)
        },
        {
            key: 'RMSE',
            label: <MetricLabel title="RMSE ($)" interpretation={interpretations.RMSE} />, // Indicar unidades ($)
            children: formatMetricValue(metrics.RMSE)
        },
        {
            key: 'MSE',
            label: <MetricLabel title="MSE ($²)" interpretation={interpretations.MSE} />, // Indicar unidades ($²)
            children: formatMetricValue(metrics.MSE)
        },
        // Puedes añadir más métricas aquí si las calculas en el backend
    ].filter(item => metrics[item.key] !== undefined && metrics[item.key] !== null); // Filtrar métricas no presentes

    return (
        <Descriptions
            title="Métricas de Rendimiento (Conjunto de Prueba)"
            bordered
            // Ajusta las columnas según el tamaño de pantalla si quieres
             column={{ xs: 1, sm: 1, md: 2, lg: 2, xl: 2 }}
            size="small" // Puede ser 'default', 'middle', 'small'
            items={items} // Pasar los items definidos
        />
    );
}

export default MetricsDisplay;