import numpy as np
import pandas as pd

# Constantes
interes_credito = 0.04*1.19
tope_dividendo_mensual = 1000000
plusvalia = 0.03
max_pie = 40000000

# Variables
valor_prestamo_range = np.linspace(125000000, 400000000, 25)  # Desde 100 millones hasta 200 millones
pie_range = np.linspace(0.1, 0.3, 3)  # Porcentajes desde 10% hasta 50%
periodos_range = np.arange(12, 360, 12)  # Desde 1 año hasta 30 años (en meses)
venta_range = np.arange(12, 60, 12)  # Desde 1 año hasta 30 años (en meses)

# Dataframe para almacenar resultados
resultados = pd.DataFrame(
    columns=["Valor_Prestamo", "Pie", "Periodos", "Venta", "VAN", "Dividendo_Mensual", "Caja_Final"]
)

# Simulación
for valor_prestamo in valor_prestamo_range:
    for pie_porc in pie_range:
        pie = valor_prestamo * pie_porc
        if pie <= max_pie:
            for periodos in periodos_range:
                for venta in venta_range:
                    if venta <= periodos:
                        cuota = (
                            (valor_prestamo - pie)
                            * (interes_credito / 12)
                        ) / (1 - (1 + interes_credito / 12) ** -periodos)
                        if cuota <= tope_dividendo_mensual:
                            flujo_caja = np.full(periodos, -cuota)
                            flujo_caja[-1] += valor_prestamo * (1 + plusvalia) ** (venta / 12)
                            caja_final = flujo_caja[-1]
                            van = np.sum(
                                flujo_caja / ((1 + interes_credito / 12) ** np.arange(1, periodos + 1))
                            )
                            nueva_fila = pd.DataFrame(
                                {
                                    "Valor_Prestamo": [valor_prestamo],
                                    "Pie": [pie_porc],
                                    "Periodos": [periodos],
                                    "Venta": [venta],
                                    "VAN": [van],
                                    "Dividendo_Mensual": [cuota],
                                    "Caja_Final": [caja_final]
                                }
                            )
                            resultados = pd.concat(
                                [resultados, nueva_fila], ignore_index=True
                            )

# Ordenar por VAN y mostrar el resultado con el máximo VAN
resultados.sort_values(by="VAN", ascending=False, inplace=True)
print(resultados.head(10)[["Valor_Prestamo", "Pie", "Periodos", "Venta", "VAN", "Dividendo_Mensual", "Caja_Final"]])


