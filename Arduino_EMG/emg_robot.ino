#include <Servo.h>  // add the servo libraries 
Servo myservo1;  // create servo object to control a servo
Servo myservo2;
Servo myservo3;
Servo myservo4;  

int pos1=80, pos2=60, pos3=130, pos4=100;  // define the variable of 4 servo angle,and assign the initial value (that is the boot posture angle value) 
int x;

void setup() 
{
  
  // boot posture 
  myservo1.write(pos1);  
  delay(1000);
  myservo2.write(pos2);
  myservo3.write(pos3);
  myservo4.write(pos4);
  delay(1500);

 Serial.setTimeout(100);

  Serial.begin(9600); //  set the baud rate to 9600
}


void loop() {
  
  myservo1.attach(A1);  // set the control pin of servo 1 to A1
  myservo2.attach(A0);  // set the control pin of servo 2 to A0
  myservo3.attach(6);   //set the control pin of servo 3 to D6
  myservo4.attach(9);   // set the control pin of servo 4 to D9
  
//  while (!Serial.available());
  x = Serial.readString().toInt();

  delay(1);  // lower the speed overall 30
  
  // claw
  zhuazi();
  //lower arm
  dabi();
}

//claw
void zhuazi()
{
    //claw
  if(x==4) // if push the left joystick to the right
  {
      for(int i=0; i<27; i++){
        delay(30);
        pos4=pos4-1;
        myservo4.write(pos4);//current angle of servo 4 subtracts 2（change the value you subtract, thus change the closed speed of claw）
        if(pos4<2)  // if pos4 value subtracts to 2, the claw in 37 degrees we have tested is closed.
        {            //（should change the value based on the fact）
          pos4=2;   // stop subtraction when reduce to 2
          }
      }
      //Serial.println(pos4);
   }
  if(x==5) //// if push the left joystick to the left 
  {
      for(int i=0; i<27; i++){
        delay(30);
        pos4=pos4+1; // current angle of servo 4 plus 8（change the value you plus, thus change the open speed of claw
        myservo4.write(pos4); // servo 4 operates the motion, the claw gradually opens.
        if(pos4>108)  // limit the largest angle when open the claw  
          {
          pos4=108;
          }
        }
  }
}
//******************************************************
 // turn
/*void zhuandong()
{
  if(x==2)  // if push the right joystick to the right
  {
    pos1=pos1-1;  //pos1 subtracts 1
    myservo1.write(pos1);  // servo 1 operates the motion, the arm turns right.
    delay(5);
    if(pos1<1)   // limit the angle when turn right
    {
      pos1=1;
    }
  }
  if(x1==3)  // if push the right joystick to the let
  {
    pos1=pos1+1;  //pos1 plus 1
    myservo1.write(pos1);  // arm turns left 
    delay(5);
    if(pos1>180)  // limit the angle when turn left 
    {
      pos1=180;
    }
  }
} */

//**********************************************************/
// lower arm
void dabi()
{
    if(x==2)  // if push the left joystick upward 
  {
      for(int i=0; i<27; i++){
        delay(30);
        pos3=pos3+1;
        myservo3.write(pos3);//current angle of servo 4 subtracts 2（change the value you subtract, thus change the closed speed of claw）
        if(pos3>130)  // if pos4 value subtracts to 2, the claw in 37 degrees we have tested is closed.
        {            //（should change the value based on the fact）
          pos3=130;   // stop subtraction when reduce to 2
          }
      }    
  }
  if(x==3)  // if push the left joystick downward
  {
      for(int i=0; i<27; i++){
        delay(30);
        pos3=pos3-1;
        myservo3.write(pos3);//current angle of servo 4 subtracts 2（change the value you subtract, thus change the closed speed of claw）
        if(pos3<35)  // if pos4 value subtracts to 2, the claw in 37 degrees we have tested is closed.
        {            //（should change the value based on the fact）
          pos3=35;   // stop subtraction when reduce to 2
        }
  }
}
}
