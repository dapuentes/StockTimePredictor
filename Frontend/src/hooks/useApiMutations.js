import { useMutation } from '@tanstack/react-query';
import { trainModel, generateForecast, getTrainingStatus } from '../services/api';
import { message } from 'antd';


export function useTrainModelMutation(options = {}) {
    return useMutation({
        mutationFn: ({ modelType, config }) => trainModel(modelType, config), // La función que llama a la API

        onMutate: () => {
            // Show initial submission message
            message.loading({ content: 'Enviando trabajo de entrenamiento...', key: 'trainSubmit', duration: 0 });
        },
        onSuccess: (jobData, variables, context) => {
            // jobData now contains { job_id, status, message } instead of training results
            const { job_id, status, message: jobMessage } = jobData;
            
            // Update message to reflect job has been queued
            message.success({ 
                content: `Trabajo de entrenamiento para ${variables.modelType.toUpperCase()} encolado con ID: ${job_id}`, 
                key: 'trainSubmit', 
                duration: 4 
            });
            
            // Call the success callback with job data - this should initiate polling in App.js
            if (options.onSuccessCallback) {
                options.onSuccessCallback(jobData, variables);
            }
        },
        onError: (error, variables, context) => {
            // Handle submission errors
            message.error({ 
                content: `Error enviando trabajo de entrenamiento: ${error.message}`, 
                key: 'trainSubmit', 
                duration: 5 
            });
            if (options.onErrorCallback) {
                options.onErrorCallback(error, variables);
            }
        },
        onSettled: (data, error, variables, context) => {
            // Ensure loading message is cleared
            message.destroy('trainSubmit');
            if (options.onSettledCallback) {
                options.onSettledCallback(data, error, variables);
            }
        },
        ...options // Permite pasar opciones adicionales al hook
    });
}


export function useGenerateForecastMutation(options = {}) {
     return useMutation({
         mutationFn: ({ modelType, config }) => generateForecast(modelType, config),
         onMutate: () => {
             message.loading({ content: 'Generando pronóstico...', key: 'forecastStatus', duration: 0 });
         },
         onSuccess: (data, variables, context) => {
             message.success({ content: 'Pronóstico generado!', key: 'forecastStatus', duration: 3 });
             if (options.onSuccessCallback) options.onSuccessCallback(data, variables);
         },
         onError: (error, variables, context) => {
             // Check if it's a 404 model not found error to provide specific messaging
             const isModelNotFoundError = error?.response?.status === 404 || 
                                        (error?.message && error.message.includes('modelo no encontrado')) ||
                                        (error?.response?.data && typeof error.response.data === 'string' && error.response.data.includes('no encontrado')) ||
                                        (error?.response?.data?.detail && error.response.data.detail.includes('no encontrado'));
             
             if (isModelNotFoundError) {
                 // For model not found errors, show a more specific message but don't use error styling
                 message.warning({ 
                     content: `Modelo no encontrado para ${variables.modelType.toUpperCase()}`, 
                     key: 'forecastStatus', 
                     duration: 3 
                 });
             } else {
                 // For other errors, show the standard error message
                 message.error({ 
                     content: `Error pronosticando: ${error.message}`, 
                     key: 'forecastStatus', 
                     duration: 5 
                 });
             }
             
             // Always pass the error and variables to the callback for App.js to handle
             if (options.onErrorCallback) {
                 options.onErrorCallback(error, variables);
             }
         },
         onSettled: (data, error, variables, context) => {
             message.destroy('forecastStatus');
             if (options.onSettledCallback) options.onSettledCallback(data, error, variables);
         },
         ...options
     });
}


export function useTrainingStatusMutation(options = {}) {
    return useMutation({
        mutationFn: ({ modelType, jobId }) => getTrainingStatus(modelType, jobId),
        
        onSuccess: (statusData, variables, context) => {
            // Handle successful status retrieval
            if (options.onSuccessCallback) {
                options.onSuccessCallback(statusData, variables);
            }
        },
        onError: (error, variables, context) => {
            // Handle status check errors
            console.error('Error checking training status:', error);
            if (options.onErrorCallback) {
                options.onErrorCallback(error, variables);
            }
        },
        onSettled: (data, error, variables, context) => {
            if (options.onSettledCallback) {
                options.onSettledCallback(data, error, variables);
            }
        },
        ...options
    });
}