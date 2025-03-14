import pandas as pd
import yfinance as yf

# Funcion para obtener datos de Nu Holdings
def getNuData():
    try:
        # Descargar datos
        df = yf.download('NU', start='2020-12-10', end=pd.Timestamp.today().strftime('%Y-%m-%d'))

        # Se deja unicamente la fecha y el cierre
        df = df.reset_index()
        df = df[['Date', 'Close']]

        # CSV
        df.to_csv('NU_Historical_Data.csv', index=False)

        print("NU_Historical_Data.csv se ha actualizado exitosamente.")
    except Exception as e:
        print(f"Error al actualizar NU_Historical_Data.csv: {e}")
    
if __name__ == '__main__':
    getNuData()