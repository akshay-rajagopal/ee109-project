package spatial.codegen.chiselgen

import argon.core._
import argon.nodes._
import spatial.aliases._
import spatial.metadata._
import spatial.nodes._

trait ChiselGenMath extends ChiselGenSRAM {

  override protected def name(s: Dyn[_]): String = s match {
    case Def(FixRandom(x)) => s"${s}_fixrnd"
    case Def(FixNeg(x:Exp[_]))  => s"""${s}_${s.name.getOrElse(s"neg${quoteOperand(x)}")}"""
    case Def(FixAdd(x:Exp[_],y:Exp[_]))  => s"""${s}_${s.name.getOrElse("sum")}"""
    case Def(FixSub(x:Exp[_],y:Exp[_]))  => s"""${s}_${s.name.getOrElse("sub")}"""
    case Def(FixDiv(x:Exp[_],y:Exp[_]))  => s"""${s}_${s.name.getOrElse("div")}"""
    case Def(FixMul(x:Exp[_],y:Exp[_]))  => s"""${s}_${s.name.getOrElse("mul")}"""
    case _ => super.name(s)
  } 

  def quoteOperand(s: Exp[_]): String = s match {
    case ss:Sym[_] => s"x${ss.id}"
    case Const(xx:Exp[_]) => s"${boundOf(xx).toInt}"
    case _ => "unk"
  }

  override protected def emitNode(lhs: Sym[_], rhs: Op[_]): Unit = rhs match {
    case FixMul(x,y) => alphaconv_register(src"$lhs"); emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})");emit(src"${lhs}.r := ($x.*-*($y, ${latencyOptionString("FixMul", Some(bitWidth(lhs.tp)))}).r)")

    case FixDiv(x,y) => emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})");emit(src"${lhs}.r := ($x./-/($y, ${latencyOptionString("FixDiv", Some(bitWidth(lhs.tp)))}).r)")

    case FixMod(x,y) => emitGlobalWireMap(src"$lhs",src"Wire(${newWire(lhs.tp)})");emit(src"$lhs := $x.%-%($y, ${latencyOptionString("FixMod", Some(bitWidth(lhs.tp)))})")

    case FixAbs(x)  => emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})");emit(src"${lhs}.r := Mux(${x} < 0.U, -$x, $x).r")

    case FltAbs(x)  => 
      val (e,g) = x.tp match {case FltPtType(g,e) => (e,g)}
      emit(src"val $lhs = Mux(${x} < 0.FlP($e,$g), -$x, $x)")
    case FltLog(x)  => x.tp match {
      case DoubleType() => emit(src"val $lhs = Math.log($x)")
      case FloatType()  => emit(src"val $lhs = Math.log($x.toDouble).toFloat")
    }
    case FltExp(x)  => x.tp match {
      case DoubleType() => emit(src"val $lhs = Math.exp($x)")
      case FloatType()  => emit(src"val $lhs = Math.exp($x.toDouble).toFloat")
    }
    case FltSqrt(x) => x.tp match {
      case DoubleType() => emit(src"val $lhs = Utils.sqrt($x)")
      case FloatType()  => emit(src"val $lhs = Utils.sqrt($x)")
    }

    case FltPow(x,y) => if (emitEn) throw new Exception("Pow not implemented in hardware yet!")
    case FixFloor(x) => emit(src"val $lhs = Utils.floor($x)")
    case FixCeil(x) => emit(src"val $lhs = Utils.ceil($x)")

    case FltSin(x)  => throw new spatial.TrigInAccelException(lhs)
    case FltCos(x)  => throw new spatial.TrigInAccelException(lhs)
    case FltTan(x)  => throw new spatial.TrigInAccelException(lhs)
    case FltSinh(x) => throw new spatial.TrigInAccelException(lhs)
    case FltCosh(x) => throw new spatial.TrigInAccelException(lhs)
    case FltTanh(x) => throw new spatial.TrigInAccelException(lhs)
    case FltAsin(x) => throw new spatial.TrigInAccelException(lhs)
    case FltAcos(x) => throw new spatial.TrigInAccelException(lhs)
    case FltAtan(x) => throw new spatial.TrigInAccelException(lhs)

    case Mux(sel, a, b) => 
      emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})")
      // lhs.tp match { 
      //   case FixPtType(s,d,f) => 
      //     emitGlobalWire(s"""val ${quote(lhs)} = Wire(new FixedPoint($s,$d,$f))""")
      //   case _ =>
      //     emitGlobalWire(s"""val ${quote(lhs)} = Wire(UInt(${bitWidth(lhs.tp)}.W))""")
      // }
      emit(src"${lhs}.r := Mux(($sel), ${a}.r, ${b}.r)")

    // Assumes < and > are defined on runtime type...
    case Min(a, b) => emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})");emit(src"${lhs}.r := Mux(($a < $b), $a, $b).r")
    case Max(a, b) => emitGlobalWireMap(src"$lhs", src"Wire(${newWire(lhs.tp)})");emit(src"${lhs}.r := Mux(($a > $b), $a, $b).r")

    case _ => super.emitNode(lhs, rhs)
  }

}