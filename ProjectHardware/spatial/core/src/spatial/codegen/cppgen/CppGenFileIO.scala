package spatial.codegen.cppgen

import argon.codegen.cppgen.CppCodegen
import argon.core._
import spatial.aliases._
import spatial.nodes._


trait CppGenFileIO extends CppCodegen  {

  override protected def emitNode(lhs: Sym[_], rhs: Op[_]): Unit = rhs match {
    case OpenFile(filename, isWr) => 
    	val dir = if (isWr) "o" else "i"
    	emit(src"""std::${dir}fstream ${lhs}_file ($filename);""")
      emit(src"""assert(${lhs}_file.good() && "File ${s"$filename".replace("\"","")} does not exist"); """)
    case CloseFile(file) =>
    	emit(src"${file}_file.close();")
    case ReadTokens(file, delim) =>
    	emit(src"std::vector<string>* ${lhs} = new std::vector<string>; ")
    	open(src"if (${file}_file.is_open()) {")
    		open(src"while ( ${file}_file.good() ) {")
	    		emit(src"string ${lhs}_line;")
  	  		emit(src"""getline (${file}_file, ${lhs}_line);""")
  	  		if (src"""${delim}""" == """"\n"""") {
		  	  		emit(src"""if (${lhs}_line != "") {${lhs}->push_back(${lhs}_line);}""")
	  			} else {
		    		emit(src"string ${lhs}_delim = ${delim};".replace("'","\""))
	  	  		emit(src"size_t ${lhs}_pos = 0;")
	  	  		open(src"while (${lhs}_line.find(${lhs}_delim) != std::string::npos | ${lhs}_line.length() > 0) {")
              open(src"if (${lhs}_line.find(${lhs}_delim) != std::string::npos) {")
                emit(src"${lhs}_pos = ${lhs}_line.find(${lhs}_delim);")
              closeopen("} else {")
                emit(src"${lhs}_pos = ${lhs}_line.length();")
              close("}")
	  	  			emit(src"string ${lhs}_token = ${lhs}_line.substr(0, ${lhs}_pos);")
	  	  			emit(src"${lhs}_line.erase(0, ${lhs}_pos + ${lhs}_delim.length());")
		  	  		emit(src"${lhs}->push_back(${lhs}_token);")
	  	  		close("}")
	  			}
  	  	close("}")
  	  close("}")
  	  emit(src"${file}_file.clear();")
			emit(src"${file}_file.seekg(0, ${file}_file.beg);")
    case WriteTokens(file, delim, len, token, i) =>
    	open(src"for (int ${i} = 0; ${i} < $len; ${i}++) {")
    		open(src"if (${file}_file.is_open()) {")
          visitBlock(token)
    			emit(src"${file}_file << ${token.result};")
	    		val chardelim = src"$delim".replace("\"","'").replace("string(","").dropRight(1)
    			emit(src"""${file}_file << ${chardelim};""")
    		close("}")
    	close("}")

    case _ => super.emitNode(lhs, rhs)
  }



}
