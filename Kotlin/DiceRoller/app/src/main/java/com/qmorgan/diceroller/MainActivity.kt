package com.qmorgan.diceroller

import android.support.v7.app.AppCompatActivity
import android.os.Bundle

fun main(){
    val dice = Dice(6)
    val dRoll = dice.roll()
    println(dRoll)
}

class Dice(val sides: Int){
    fun roll(): Int{
        val roll = (1..sides).random()
        return roll
    }
}