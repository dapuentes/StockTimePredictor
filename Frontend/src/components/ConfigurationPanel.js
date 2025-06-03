import React, { useState, useEffect } from 'react';
import {
    Form,
    Select,
    DatePicker,
    InputNumber,
    Button,
    Divider,
    Typography
} from 'antd';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Option } = Select;
const { Title } = Typography;

// Convertir Date a dayjs y viceversa para el RangePicker
const dateToDayjs = (date) => (date ? dayjs(date) : null);
const dayjsToDate = (dayjsObj) => (dayjsObj ? dayjsObj.toDate() : null);


function ConfigurationPanel({
    availableModelTypes,
    availableTickers,
    onConfigChange,
    onTrain,
    onForecast,
    initialConfig
}) {
    // Estado local para manejar los valores del formulario
    // Lo inicializamos desde initialConfig para que refleje el estado de App.js
    const [formValues, setFormValues] = useState({
         ...initialConfig,
         dateRange: [dateToDayjs(initialConfig.startDate), dateToDayjs(initialConfig.endDate)],
     });

     useEffect(() => {
         setFormValues({
             ...initialConfig,
             dateRange: [dateToDayjs(initialConfig.startDate), dateToDayjs(initialConfig.endDate)],
         });
     }, [initialConfig]);

    const handleValueChange = (changedValues) => {
        const newValues = { ...formValues, ...changedValues };

        // Si cambió el dateRange de dayjs, convertirlo a Date para el estado global
        if ('dateRange' in changedValues) {
            const [startDayjs, endDayjs] = changedValues.dateRange || [null, null];
            newValues.startDate = dayjsToDate(startDayjs);
            newValues.endDate = dayjsToDate(endDayjs);
        }

        setFormValues(newValues); // Actualizar estado local
        onConfigChange(newValues); // Notificar al padre con el objeto completo actualizado
    };

    // Manejador para el botón de Entrenar
    const handleTrainClick = () => {
        onTrain(formValues.selectedModelType, formValues);
    };

    // Manejador para el botón de Pronosticar
     const handleForecastClick = () => {
         onForecast(formValues.selectedModelType, formValues);
     };

    // Layout para los items del formulario (label arriba, input abajo)
    const formItemLayout = {
        labelCol: { span: 24 },
        wrapperCol: { span: 24 },
    };

    return (
        <Form
            {...formItemLayout}
             layout="vertical" // Labels arriba
             onValuesChange={handleValueChange} // Llama a handleValueChange cuando cualquier campo cambia
             initialValues={{ // Sincronizar valores iniciales
                 selectedTicker: formValues.selectedTicker,
                 dateRange: formValues.dateRange,
                 selectedModelType: formValues.selectedModelType,
                 nLags: formValues.nLags,
                 forecastHorizon: formValues.forecastHorizon
             }}
        >
            {/* Selección de Ticker */}
            <Form.Item label="Acción (Ticker)" name="selectedTicker">
                <Select placeholder="Selecciona un ticker">
                    {availableTickers.map(ticker => <Option key={ticker} value={ticker}>{ticker}</Option>)}
                </Select>
            </Form.Item>

            {/* Selección de Rango de Fechas */}
            <Form.Item label="Rango Fechas (Entrenamiento)" name="dateRange">
                <RangePicker style={{ width: '100%' }} format="YYYY-MM-DD" />
            </Form.Item>

            {/* Selección de Modelo */}
            <Form.Item label="Tipo de Modelo" name="selectedModelType">
                 <Select placeholder="Selecciona un tipo de modelo">
                     {availableModelTypes.map(type => <Option key={type} value={type}>{type.toUpperCase()}</Option>)}
                 </Select>
            </Form.Item>



            {/* Número de Lags */}
            <Form.Item label="Número de Lags (Entrenamiento)" name="nLags">
                <InputNumber min={1} step={1} style={{ width: '100%' }} />
            </Form.Item>

            {/* Botón de Entrenamiento */}
            <Form.Item> {/* Wrapper para el botón */}
                <Button
                    type="primary" // Estilo principal
                    onClick={handleTrainClick}
                    block // Botón
                    disabled={!formValues.selectedTicker || !formValues.selectedModelType} // Deshabilitar si falta algo
                >
                    Entrenar Modelo ({formValues.selectedModelType?.toUpperCase()})
                </Button>
            </Form.Item>

            <Divider /> {/* Línea divisoria */}

            <Title level={4}>Pronóstico</Title> {/* Título para la sección */}

            {/* Horizonte de Pronóstico */}
            <Form.Item label="Horizonte de Pronóstico (días)" name="forecastHorizon">
                <InputNumber min={1} step={1} style={{ width: '100%' }} />
            </Form.Item>

            {/* Botón para Generar Pronóstico */}
             <Form.Item>
                 <Button
                    type="primary"
                    onClick={handleForecastClick}
                    block
                    disabled={!formValues.selectedTicker || !formValues.selectedModelType}
                 >
                    Generar Pronóstico ({formValues.selectedModelType?.toUpperCase()})
                </Button>
            </Form.Item>
        </Form>
    );
}

export default ConfigurationPanel;