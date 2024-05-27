use logos::Logos;
use swell::lexer::TokenKind;

fn main() {
    let input = "let    abCDEFG = 42";
    let source_file = "example.txt";
    let lexer = swell::lexer::Lexer::new(input);

    let testLexer = TokenKind::lexer(input);

    for val in testLexer.enumerate() {
        println!("{:?}", val)
    }
}
