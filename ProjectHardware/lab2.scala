import spatial.dsl._
import org.virtualized._


// MemReduce
object Lab2Part1SimpleMemReduce extends SpatialApp {

  val N = 16.to[Int]

  @virtualize
  def main() {
    val out = DRAM[Int](16)
    Accel {
      val a = SRAM[Int](16)
      MemReduce(a)(-5 until 5 by 1){i =>
        val tmp = SRAM[Int](16)
        Foreach(16 by 1) { j => tmp(j) = 1}
        tmp
      }{_+_}
      out store a
    }

    val result = getMem(out)
    val gold = Array.tabulate(16){i => 10.to[Int]}
    printArray(gold, "expected: ")
    printArray(result, "result:   ")

    val cksum = gold.zip(result){_==_}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part1SimpleMemReduce)")
  }
}


// MemFold
object Lab2Part2SimpleMemFold extends SpatialApp {

  val N = 16.to[Int]

  @virtualize
  def main() {
    val out = DRAM[Int](16)
    Accel {
      val a = SRAM[Int](16)
      Foreach(16 by 1) { j => a(j) = 0}
      MemFold(a)(-5 until 5 by 1){i =>
        val tmp = SRAM[Int](16)
        Foreach(16 by 1) { j => tmp(j) = 1}
        tmp
      }{_+_}
      out store a
    }

    val result = getMem(out)
    val gold = Array.tabulate(16){i => 10.to[Int]}
    printArray(gold, "expected: ")
    printArray(result, "result:   ")

    val cksum = gold.zip(result){_==_}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part2SimpleMemFold)")
  }
}


// FSM
object Lab2Part3BasicCondFSM extends SpatialApp { // Regression (Unit) // Args: none


  @virtualize
  def main() {
    val dram = DRAM[Int](32)
    Accel {
      val bram = SRAM[Int](32)
      val reg = Reg[Int](0)
      reg := 16
      FSM[Int]{state => state < 32} { state =>
        if (state < 16) {
          if (state < 8) {
            bram(31 - state) = state // 16:31 [7, 6, ... 0]
          } else {
            bram(31 - state) = state+1 // 16:31 [16, 15, ... 9]
          }
        }
        else {
          bram(state - 16) = if (state == 16) 17 else if (state == 17) reg.value else state // Test const, regread, and bound Mux1H
        }
      }{state => state + 1}

      dram(0::32) store bram
    }
    val result = getMem(dram)
    val gold = Array[Int](17, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31, 16, 15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 1, 0)
    printArray(result, "Result")
    printArray(gold, "Gold")
    val cksum = gold.zip(result){_ == _}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part3BasicCondFSM)")
  }
}



// FSMAlt
object Lab2Part3BasicCondFSMAlt extends SpatialApp {

  @virtualize
  def main() {
    val dram = DRAM[Int](32)
    Accel {
      val bram = SRAM[Int](32)
      val reg = Reg[Int](0)
      reg := 16
      FSM[Int]{state => state < 32} { state =>
        if (state < 8) {
            bram(state) = state 
        } else if (state < 16) {
            bram(state) = state*2 
        } else if (state < 24) {
            bram(state) = state*3
        } else {
          bram(state) = state*4
        }
      }{state => state + 1}

      dram(0::32) store bram      
    }
    val result = getMem(dram)
    val gold = Array[Int](0, 1, 2, 3, 4, 5, 6, 7, 16, 18, 20, 22, 24,
                          26, 28, 30, 48, 51, 54, 57, 60, 63, 66, 69,
                          96, 100, 104, 108, 112, 116, 120, 124)
    printArray(result, "Result")
    printArray(gold, "Gold")
    val cksum = gold.zip(result){_ == _}.reduce{_&&_}
    println("PASS: " + cksum + " (Lab2Part3BasicCondFSMAlt)")
  }
}


object Lab2Part4LUT extends SpatialApp {
  @virtualize
  def main() {
    type T = Int
    val M = 3
    val N = 3

    val in = ArgIn[T]
    val out = ArgOut[T]
    val i = ArgIn[T]
    val j = ArgIn[T]

    val input = args(0).to[T]
    val ind_i = args(1).to[T]
    val ind_j = args(2).to[T]

    setArg(in, input)
    setArg(i, ind_i)
    setArg(j, ind_j)

    Accel {
      // Your code here
      val inVal = in.value
      val iVal = i.value
      val jVal = j.value
      
      val lut = LUT[T](3,3)(1,2,3,4,5,6,7,8,9)
      val lut_ij = lut(iVal,jVal)
      out := inVal + lut_ij
    }

    val result = getArg(out)
    val goldArray = Array.tabulate(M * N){ i => i + 1 }
    val gold = input + goldArray(i*N + j)
    val pass = gold == result
    println("PASS: " + pass + "(Lab2Part4LUT)")
  }
}


// GEMM
object Lab2Part5GEMM extends SpatialApp {

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 2

    type T = FixPt[TRUE,_24,_8]
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val M = ArgIn[Int]
    val N = ArgIn[Int]
    val K = ArgIn[Int]
    setArg(M,args(0).to[Int])
    setArg(N,args(1).to[Int])
    setArg(K,args(2).to[Int])

    val a_data = (0::args(0).to[Int], 0::args(2).to[Int]){(i,j) => random[T](3)}
    val b_data = (0::args(2).to[Int], 0::args(1).to[Int]){(i,j) => random[T](3)}
    val c_init = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) => 0.to[T]}
    val a = DRAM[T](M, K)
    val b = DRAM[T](K, N)
    val c = DRAM[T](M, N)

    setMem(a, a_data)
    setMem(b, b_data)
    setMem(c, c_init)

    Accel {
      Foreach(K by tileK par outerPar){kk =>
        val numel_k = min(tileK.to[Int], K - kk)
        Foreach(M by tileM par innerPar){mm =>
          val numel_m = min(tileM.to[Int], M - mm)
          val tileA_sram = SRAM[T](tileM, tileK)
          tileA_sram load a(mm::mm+numel_m, kk::kk+numel_k)
          Foreach(N by tileN par innerPar){nn =>
            val numel_n = min(tileN.to[Int], N - nn)
            val tileB_sram = SRAM[T](tileK, tileN)
            val tileC_sram = SRAM.buffer[T](tileM, tileN)
            tileB_sram load b(kk::kk+numel_k, nn::nn+numel_n)
            tileC_sram load c(mm::mm+numel_m, nn::nn+numel_n)

            // Your code here
            MemFold(tileC_sram)(0 until numel_k by 1){k =>
              val tmp = SRAM[T](tileM,tileN)
              Foreach(0 until numel_m by 1 par tileM) { i => 
                Foreach(0 until numel_n by 1 par tileN) {j => 
                  tmp(i,j) = tileA_sram(i,k)*tileB_sram(k,j)
                }
              }
              tmp
            }{_+_}
            c(mm::mm+numel_m, nn::nn+numel_n) store tileC_sram
          }
        }
      }
    }

    val accel_matrix = getMatrix(c)
    val gold_matrix = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) =>
      Array.tabulate(args(2).to[Int]){k => a_data(i,k) * b_data(k,j)}.reduce{_+_}
    }

    printMatrix(accel_matrix, "Received: ")
    printMatrix(gold_matrix, "Wanted: ")
    val cksum = accel_matrix.zip(gold_matrix){_==_}.reduce{_&&_}
    println("PASS: " + cksum + "(Lab2Part5GEMM)")
  }
}

// Digits SVM
object ProjectSVM extends SpatialApp {

  @virtualize
  def main() {

    val outerPar = 1
    val midPar = 2
    val innerPar = 2

    type T = FixPt[TRUE,_24,_8]
    val tileM = 16
    val tileN = 16
    val tileK = 16

    val picSize = 400
    val numTrainImages = 60000.to[Int]
    val numTestImages = 10000.to[Int]
    val digits = 10.to[Int]

    // val M = ArgIn[Int]
    // val N = ArgIn[Int]
    // val K = ArgIn[Int]
    // setArg(M,args(0).to[Int])
    // setArg(N,args(1).to[Int])
    // setArg(K,args(2).to[Int])

    // val rho = 0.25.to[T]
    val rho = 0.04.to[T]
    val base_alpha = 0.0001.to[T]

    val train_data = (0::picSize, 0::numTrainImages){(i,j) => random[T](3)}
    val train_labels = (0::numTrainImages){i => 0.to[T]}
    val test_data = (0::picSize, 0::numTestImages){(i,j) => random[T](3)}
    val test_labels = (0::numTestImages){i => 0.to[T]}
    val W_init = (0::picSize, 0::digits){(i,j) => 0.to[T]}
    val trainImages = DRAM[T](picSize, numTrainImages)
    val trainLabels = DRAM[T](numTrainImages)
    val testImages = DRAM[T](picSize, numTestImages)
    val testLabels = DRAM[T](numTestImages)
    val W = DRAM[T](picSize, digits)

    setMem(trainImages, train_data)
    setMem(trainLabels, train_labels)
    setMem(testImages, test_data)
    setMem(testLabels, test_labels)
    setMem(W, W_init)

    Accel {
      Foreach(trainImages by 1){k =>
        val img_sram = SRAM[T](picSize)
        img_sram load trainImages(k*picSize::(k+1)*picSize)
        val label = trainLabels(k)
        Foreach(digits by 1) {i =>
          val y = mux(i == label, 1, -1)
          //val alpha = mux(i == label, 10/k, 2/k)
          val alpha = mux(i == label, 5*base_alpha, base_alpha)
          val ywx = Reg[T](0)
          Reduce(accum)(picSize by 1) {j
            val el = y * W(i,j) * img_sram(j)
          }{_ + _}
          val gk1_sram = SRAM[T](picSize)
          Foreach(picSize by 1){j =>
            gk1_sram(j) = mux(select > 0, rho * W(i,j) + y * img_sram(j),rho * W(i,j))
          }
          val select = 1 - accum
          Foreach(picSize by 1){j =>
            W(i,j) = W(i,j) - alpha * gk1_sram(j)
          }
        }

      val errors = Reg[Int](0)
      Foreach(testImages by 1){k =>
        val img_sram = SRAM[T](picSize)
        img_sram load testImages(k*picSize::(k+1)*picSize)
        val label = testLabels(k)
        val maxind = Reg[T](0)
        Reduce(maxind)(digits by 1){ i =>
          val res = Reg[T](0)
          Reduce(res)(picSize by 1){j =>
            W(i,j) * img_sram(j)
          }{_ + _}
        }{}
        errors := mux(maxind == label, errors, errors + 1)
      }

    }

    val accel_matrix = getMatrix(W)
    //val gold_matrix = (0::args(0).to[Int], 0::args(1).to[Int]){(i,j) =>
    //  Array.tabulate(args(2).to[Int]){k => a_data(i,k) * b_data(k,j)}.reduce{_+_}
    //}

    //printMatrix(accel_matrix, "Received: ")
    //printMatrix(gold_matrix, "Wanted: ")
    //val cksum = accel_matrix.zip(gold_matrix){_==_}.reduce{_&&_}
    //println("PASS: " + cksum + "(Lab2Part5GEMM)")
  }
}