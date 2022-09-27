void setup() 
{
    // Бодрейт 9600 
    Serial.begin(9600);
}

//На каждой итерации производится 
// сбор данных с ЭМГ-датчика и 
// отправка этих данных в последовательный порт:
void loop()
{

    int val = analogRead(A0); // get Analog value
    
    Serial.println(val);
    
    delay(10);
}
