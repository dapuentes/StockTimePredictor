import React, { useRef } from 'react';
import { Line } from 'react-chartjs-2';
import { Button } from 'antd';
import { CameraOutlined, ReloadOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns'; // Adaptador para fechas
import zoomPlugin from 'chartjs-plugin-zoom'; // Plugin de zoom y pan


ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale, // Registrar escala de tiempo
  zoomPlugin // Registrar plugin de zoom
);

/**
 * Displays a line chart combining historical and forecast data for a specific stock ticker.
 *
 * @param {Object} params An object containing the historical data, forecast data, and the ticker symbol.
 * @param {Object} params.historicalData Historical data for the stock. Should include `dates` (array of strings) and `values` (array of numbers).
 * @param {Array<Object>} params.forecastData Forecast data for the stock. Each object should contain `date` (string) and `prediction` (number).
 * @param {string} params.ticker The stock ticker symbol.
 * @return {JSX.Element} Returns a line chart JSX component displaying the combined historical and forecast data. If no data is available, a message is rendered prompting the user to generate a forecast.
 */
function GraphDisplay({ historicalData, forecastData, ticker }) {
    const chartRef = useRef(null);

    const hasHistoricalData = historicalData?.dates?.length > 0 && historicalData?.values?.length > 0;

    const hasForecastData = forecastData?.length > 0;

    if (!hasHistoricalData && !hasForecastData) {

        return <p style={{ textAlign: 'center', padding: '20px' }}>Genera un pronóstico para ver la gráfica.</p>;
    }


    const historicalDates = hasHistoricalData ? historicalData.dates : [];
    const forecastDates = hasForecastData ? forecastData.map(p => p.date) : [];

    const allDates = [...new Set([...historicalDates, ...forecastDates])].sort((a, b) => new Date(a) - new Date(b));

    const historicalMap = new Map(hasHistoricalData ? historicalData.dates.map((d, i) => [d, historicalData.values[i]]) : []);
    const forecastMap = new Map(hasForecastData ? forecastData.map(p => [p.date, p.prediction]) : []);

    const historicalMappedValues = allDates.map(date => historicalMap.get(date) ?? null);

    const forecastMappedValues = allDates.map(date => forecastMap.get(date) ?? null);

    const chartData = {
        labels: allDates, // Use the combined date array as labels
        datasets: []
    };



    // Add historical dataset using the mapped values
    if (hasHistoricalData) {
         chartData.datasets.push({
            label: `Precio Histórico (${ticker})`,
            data: historicalMappedValues, // Use the mapped array with nulls
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.1,
            pointRadius: 1, // Puntos más pequeños para histórico
            pointHitRadius: 10,
            fill: false
        });
    }

    // Add forecast dataset using the mapped values
    if (hasForecastData) {
        chartData.datasets.push({
            label: 'Pronóstico',
            data: forecastMappedValues,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            borderDash: [5, 5], // Línea punteada
            tension: 0.1,
            pointRadius: 2,
            pointHitRadius: 10,
            fill: false
        });
    }

    const handleResetZoom = () => {
        if (chartRef.current) {
            chartRef.current.resetZoom();
        }
    }

    const handleExportImage = () => {
        if (chartRef.current) {
            const imageBase64 = chartRef.current.toBase64Image();
            const link = document.createElement('a');
            link.href = imageBase64;
            const filename = `grafico_${ticker}_${dayjs().format('YYYYMMDD_HHmmss')}.png`;
            link.download = filename;
            link.click();
        }
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: `Serie de Tiempo y Pronóstico para ${ticker}` },
            tooltip: {
                mode: 'index',
                intersect: false,
                filter: function(tooltipItem) {
                   return tooltipItem.raw !== null;
                }
            },
            zoom: {
                pan: {
                    enabled: true, // Habilitar paneo (mover el gráfico)
                    mode: 'xy',
                    threshold: 5, // Píxeles a mover antes de iniciar paneo
                },
                zoom: {
                    wheel: {
                        enabled: true, // Habilitar zoom con rueda del ratón
                    },
                    pinch: {
                        enabled: true // Habilitar zoom con "pellizco" en táctiles
                    },
                    mode: 'xy',
                },
                limits: {
                    x: {min: 'original', max: 'original'},
                    y: {min: 'original', max: 'original'}
                }
            }
        },
        scales: {
        },
        interaction: {
            mode: 'index',
            intersect: false
        },
    };

    // Contenedor con altura definida es importante
    return (
        <div style={{ height: '450px', width: '100%', position: 'relative' }}>
            <Button
                icon={<ReloadOutlined />}
                onClick={handleResetZoom}
                style={{ position: 'absolute', top: '20px', right: '10px', zIndex: 10 }}
                size="small"
                title="Restablecer Zoom"
             />
            <Button
                icon={<CameraOutlined />}
                onClick={handleExportImage}
                style={{ position: 'absolute', top: '20px', right: '50px', zIndex: 10 }}
                size="small"
                title="Exportar Imagen"
            />
             <Line
                ref={chartRef}
                options={options}
                data={chartData}
                key={ticker + (historicalData?.dates?.length || 0) + (forecastData?.length || 0)}
             />
        </div>
    );

}

export default GraphDisplay;