import os


def part_1():
    os.system("python data_analysis.py")


def part_2_classification():
    os.system("python classification.py")


def part_2_classification_sklearn():
    os.system("python classification_sklearn.py")


def part_2_models_comparison():
    os.system("python models_comparison.py")


def part_3():
    os.system("python unsupervised.py")


def main():
    while True:
        print("\nSelecciona una parte del proyecto:")
        print("1. Parte 1: Análisis y división de los datos")
        print("2. Parte 2: Modelos de clasificación")
        print("3. Parte 3: Modelos no supervisados")
        print("0. Salir")

        opcion = input(
            "\nIngresa el número de la parte que deseas ejecutar (0 para salir): "
        )

        if opcion == "1":
            part_1()
        elif opcion == "2":
            while True:
                print("\nSelecciona una opción de la parte 2:")
                print("1. Modelos de clasificación")
                print("2. Modelos de clasificación con sklearn")
                print("3. Comparativa de modelos")
                print("0. Volver al menú principal")

                opcion_parte_2 = input(
                    "\nIngresa el número de la opción que deseas ejecutar (0 para volver al menú principal): "
                )

                if opcion_parte_2 == "1":
                    part_2_classification()
                elif opcion_parte_2 == "2":
                    part_2_classification_sklearn()
                elif opcion_parte_2 == "3":
                    part_2_models_comparison()
                elif opcion_parte_2 == "0":
                    break
                else:
                    print("Opción no válida. Por favor, intenta de nuevo.")
        elif opcion == "3":
            part_3()
        elif opcion == "0":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")


if __name__ == "__main__":
    main()
