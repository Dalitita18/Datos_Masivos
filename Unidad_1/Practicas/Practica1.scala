// Assessment 1/Practica 1
//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"
//5. Cual es la diferencia en value y una variable en scala?
//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 


//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
val deametro =4.5
val radio = deametro/2

//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
def primo (i : Int) : Unit =
 {
    if (i%2 != 0)
    {
        println("Es primo")
    }
    else 
    {
        println("No es primo")
    }
 }

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy ecribiendo un tweet"
val bird = "tweet"
printf("Estoy ecribiendo un %s", bird)

//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke"
val mensaje = "Hola Luke yo soy tu padre!" 
mensaje slice (5,9)

//5. Cual es la diferencia en value y una variable en scala?
// Al utilizar Value, esta no puede ser modificada
// El utilizar Variable, esta si puede ser modificada

//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416 
val tupla = (2,4,5,1,2,3,3.1416,23)
tupla._7