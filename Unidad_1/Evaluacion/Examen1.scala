//Villanueva Langarica Elva Dali 
//16211348

def breakingRecords (nums:List[Int]) : Unit =
{
    var minimo =nums(0)
    var maximo =nums(0)
    var contadorminimo = 0
    var contadormaximo =0
    for (i <- nums)
    {
        if (i<minimo)
        {
            minimo = i
            contadorminimo= contadorminimo +1
        }
        if (i>maximo)
        {
            maximo = i
            contadormaximo = contadormaximo + 1
        }
    }
    println (contadormaximo,contadorminimo)
}
var lista = List(10,5,20,20,4, 5, 2, 25, 1)
var lista2 = List(3,4,21,36,10,28,35,5,24,42)
breakingRecords(lista)
breakingRecords(lista2)