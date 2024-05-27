use crate::lexer::NodeKind::*;
use crate::lexer::TokenKind::*;
use logos::{Logos, Source};
use std::cell::Cell;
use std::cmp::PartialEq;
use std::fmt;
use std::ops::{Deref, Range};

pub type Span = Range<usize>;

pub struct State {
    pub span: Span,
}

impl Default for State {
    fn default() -> Self {
        State {
            span: Span::default(),
        }
    }
}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(extras = State)]
pub enum TokenKind {
    #[token("var")]
    VarKeyword,

    #[token("let")]
    LetKeyword,

    #[token("class")]
    ClassKeyword,
    #[token("fn")]
    FunctionKeyword,

    #[token("pub")]
    PublicKeyword,

    #[token("internal")]
    InternalKeyword,

    #[token("static")]
    StaticKeyword,

    #[token("protected")]
    ProtectedKeyword,

    #[token("if")]
    IfKeyword,
    #[token("new")]
    NewKeyword,
    #[token("return")]
    ReturnKeyword,
    #[token("set")]
    SetKeyword,
    #[token("get")]
    GetKeyword,
    #[token("impl")]
    ImplKeyword,
    #[token("interface")]
    InterfaceKeyword,
    #[token("true")]
    TrueKeyword,
    #[token("false")]
    FalseKeyword,
    #[token("in")]
    InKeyword,
    #[token("prop")]
    PropertyKeyword,
    #[token("auto")]
    AutoKeyword,

    #[token("=")]
    Equals,

    #[token("\"")]
    DoubleQuote,

    #[token("'")]
    SingleQuote,

    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token("{")]
    LeftCurley,

    #[token("}")]
    RightCurley,

    #[token("*")]
    Star,
    #[token("/")]
    ForwardSlash,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,

    #[token("@")]
    At,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,
    #[token("::")]
    DoubleColon,

    #[token(";")]
    Semicolon,

    #[token("->")]
    RightDashArrow,

    #[token("=>")]
    RightEqualsArrow,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,

    #[regex(r"0[bB][01]+")]
    BinaryLiteral,
    #[regex(r"0[xX][0-9a-fA-F]+")]
    HexLiteral,

    #[regex(r"[0-9]+")]
    IntegerNumericValue,
    #[regex(r"[0-9]+[uU]")]
    UnsignedIntegerLiteral,
    #[regex(r"[0-9]+[lL]")]
    LongIntegerLiteral,

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?")]
    FloatingPointNumeric,
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?[fF]")]
    FloatLiteral,
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?[mM]")]
    DecimalLiteral,

    #[regex(r"[0-9]+[uU][lL]|[0-9]+[lL][uU]")]
    UnsignedLongIntegerLiteral,

    #[regex(r"\s+")]
    Whitespace,

    Eof,

    ErrorKind,
}

#[derive(Debug)]
#[rustfmt::skip]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub text: &'a str,
    pub source: Option<&'a str>,
    pub span: Span,
}

impl<'a> Token<'a> {
    pub(crate) fn is_whitespace(&self) -> bool {
        self.kind == Whitespace
    }
}

fn lex<'a>(text: &'a str) -> Vec<Token<'a>> {
    let mut lexer = TokenKind::lexer(text).into_iter();
    let mut result: Vec<Token<'a>> = Vec::new();
    let mut trivia: Vec<Token<'a>> = Vec::new();
    loop {
        match lexer.next() {
            Some(token) => match token {
                Ok(kind) => match kind {
                    Whitespace => trivia.push(Token {
                        kind,
                        text: lexer.source().slice(lexer.span()).unwrap(),
                        source: None,
                        span: lexer.span(),
                    }),
                    _ => {
                        result.push(Token {
                            kind,
                            text: lexer.source().slice(lexer.span()).unwrap(),
                            source: None,
                            span: lexer.span(),
                        });
                    }
                },
                Err(err) => {
                    result.push(Token {
                        kind: ErrorKind,
                        text: "",
                        source: None,
                        span: Span::default(),
                    });
                }
            },
            None => {
                break;
            }
        }
    }

    result
}

pub enum Child<'a> {
    Token(Token<'a>),
    Node(Node<'a>),
}

struct MarkOpened {
    index: usize,
}

struct MarkClosed {
    index: usize,
}

#[derive(Debug)]
pub enum NodeKind {
    ErrorNode,
    CompilationUnit,
    ClassNode,

    Function,
    TypeExpression,
    ParamList,
    Param,
    VarStatement,

    TypeExpr,

    ExprStmt,
    ExprCall,
    ExprBinary,
    ExprLiteral,
    ExprName,
    ExprParen,

    LetStmt,
    ReturnStmt,

    Arg,
    ArgList,
    Field,

    Block,
    AccessModifier,
    Property,
}

pub struct Node<'a> {
    pub kind: NodeKind,
    pub children: Vec<Child<'a>>,
}

pub struct Lexer<'a> {
    pub lexer: logos::Lexer<'a, TokenKind>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Lexer<'a> {
        return Lexer {
            lexer: TokenKind::lexer(source),
        };
    }
}

pub struct Parser<'a> {
    tokens: Vec<Token<'a>>,
    pos: usize,
    fuel: Cell<u32>,
    events: Vec<Event>,
}

#[derive(Debug)]
enum Event {
    Open { kind: NodeKind },
    Close,
    Advance,
}
impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token<'a>>) -> Parser<'a> {
        Parser {
            tokens,
            pos: 0,
            fuel: Cell::new(256),
            events: Vec::new(),
        }
    }

    fn build_node(self) -> Node<'a> {
        let mut tokens = self.tokens.into_iter();
        let mut events = self.events;

        assert!(matches!(events.pop(), Some(Event::Close)));
        let mut stack = Vec::new();
        for event in events {
            match event {
                Event::Open { kind } => stack.push(Node {
                    kind,
                    children: Vec::new(),
                }),
                Event::Close => {
                    let node = stack.pop().unwrap();
                    stack.last_mut().unwrap().children.push(Child::Node(node));
                }
                Event::Advance => {
                    let token = tokens.next().unwrap();
                    stack.last_mut().unwrap().children.push(Child::Token(token));
                }
            }
        }

        let node = stack.pop().unwrap();
        assert!(stack.is_empty());
        assert!(tokens.next().is_none());
        node
    }

    fn open(&mut self) -> MarkOpened {
        let mark = MarkOpened {
            index: self.events.len(),
        };
        self.events.push(Event::Open {
            kind: NodeKind::ErrorNode,
        });
        mark
    }

    fn open_before(&mut self, m: MarkClosed) -> MarkOpened {
        let mark = MarkOpened { index: m.index };
        self.events.insert(m.index, Event::Open { kind: ErrorNode });
        mark
    }

    fn close(&mut self, m: MarkOpened, kind: NodeKind) -> MarkClosed {
        self.events[m.index] = Event::Open { kind };
        self.events.push(Event::Close);
        MarkClosed { index: m.index }
    }

    fn advance(&mut self) {
        assert!(!self.eof());
        self.fuel.set(256);
        self.events.push(Event::Advance);
        self.pos += 1;
    }

    fn advance_with_error(&mut self, error: &str) {
        let m = self.open();
        // TODO: Error reporting.
        eprintln!("{error}");
        self.advance();
        self.close(m, ErrorNode);
    }

    fn eof(&self) -> bool {
        self.pos == self.tokens.len()
    }

    fn nth(&self, lookahead: usize) -> TokenKind {
        if self.fuel.get() == 0 {
            panic!("parser is stuck")
        }

        let mut pos = self.pos;
        let mut count = 0;
        self.fuel.set(self.fuel.get() - 1);
        while let Some(token) = self.tokens.get(pos + lookahead) {
            if !token.is_whitespace() {
                if count == lookahead {
                    return token.kind;
                }
            }
            pos += 1;
        }
        Eof
    }

    fn nth_whitespace(&self, lookahead: usize) -> TokenKind {
        if self.fuel.get() == 0 {
            panic!("parser is stuck")
        }

        let mut pos = self.pos;
        let mut count = 0;
        self.fuel.set(self.fuel.get() - 1);
        while let Some(token) = self.tokens.get(pos + lookahead) {
            if count == lookahead {
                return token.kind;
            }
            pos += 1;
        }
        Eof
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.nth(0) == kind
    }

    fn at_whitespace(&self, kind: TokenKind) -> bool {
        self.nth(0) == kind
    }

    fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.contains(&self.nth(0))
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) {
        if self.eat(kind) {
            return;
        }
        // TODO: Error reporting.
        eprintln!("expected {kind:?}");
    }
}

fn compilation_unit(p: &mut Parser) {
    let m = p.open();
    while !p.eof() {
        if p.at(ClassKeyword) || p.at_any(ACCESS_MODIFIERS) {
            class(p)
        } else {
            p.advance_with_error("expected a class");
        }
    }
    p.close(m, CompilationUnit);
}

fn class(p: &mut Parser) {
    let m = p.open();

    if p.at(ClassKeyword) {
        p.expect(ClassKeyword);
    } else if p.at_any(ACCESS_MODIFIERS) {
        access_modifier(p);
        p.expect(ClassKeyword);
    }

    p.expect(Identifier);
    p.expect(LeftCurley);

    while !p.at(RightCurley) && !p.eof() {
        if p.at_any(ACCESS_MODIFIERS) {
            access_modifier(p);
        } else if p.nth(0) == FunctionKeyword {
            function(p);
        } else if p.nth(0) == PropertyKeyword {
            property(p);
        } else if p.nth(0) == Identifier {
            field(p);
        } else {
            p.advance_with_error("Expected a curly to end the class.")
        }
    }

    p.expect(RightCurley);
    p.close(m, ClassNode);
}

fn function(p: &mut Parser) {
    assert!(p.at(FunctionKeyword));
    let m = p.open();
    p.expect(FunctionKeyword);
    p.expect(Identifier);
    if p.at(LeftParen) {
        param_list(p);
    }
    if p.eat(RightDashArrow) {
        type_expr(p);
    }
    if p.at(LeftCurley) {
        function_block(p);
    }
    p.close(m, Function);
}

fn property(p: &mut Parser) {
    assert!(p.at(PropertyKeyword));

    let m = p.open();

    p.expect(PropertyKeyword);
    p.expect(Identifier);
    p.expect(Colon);
    type_expr(p);

    if p.eat(Equals) {
        expr(p);
        p.expect(Semicolon);
    } else if p.at(Semicolon) {
        p.expect(Semicolon);
    } else if p.at(LeftCurley) {
        function_block(p);
    }

    p.close(m, Property);
}

const PARAM_LIST_RECOVERY: &[TokenKind] = &[FunctionKeyword, LeftCurley];
fn param_list(p: &mut Parser) {
    assert!(p.at(LeftParen));
    let m = p.open();

    p.expect(LeftParen);
    while !p.at(RightParen) && !p.eof() {
        if p.at(Identifier) {
            param(p);
        } else {
            if p.at_any(PARAM_LIST_RECOVERY) {
                break;
            }
            p.advance_with_error("expected parameter");
        }
    }
    p.expect(RightParen);

    p.close(m, ParamList);
}

fn param(p: &mut Parser) {
    assert!(p.at(Identifier));
    let m = p.open();

    p.expect(Identifier);
    p.expect(Colon);
    type_expr(p);
    if !p.at(RightParen) {
        p.expect(Comma);
    }

    p.close(m, Param);
}

fn type_expr(p: &mut Parser) {
    let m = p.open();
    p.expect(Identifier);
    p.close(m, TypeExpr);
}

const STMT_RECOVERY: &[TokenKind] = &[FunctionKeyword];
const EXPR_FIRST: &[TokenKind] = &[
    IntegerNumericValue,
    UnsignedIntegerLiteral,
    LongIntegerLiteral,
    FloatingPointNumeric,
    FloatLiteral,
    DecimalLiteral,
    TrueKeyword,
    FalseKeyword,
    Identifier,
    RightParen,
];
fn function_block(p: &mut Parser) {
    assert!(p.at(LeftCurley));
    let m = p.open();

    p.expect(LeftCurley);
    while !p.at(RightCurley) && !p.eof() {
        match p.nth(0) {
            LetKeyword => stmt_let(p),
            ReturnKeyword => stmt_return(p),
            _ => {
                if p.at_any(EXPR_FIRST) {
                    stmt_expr(p)
                } else {
                    if p.at_any(STMT_RECOVERY) {
                        break;
                    }
                    p.advance_with_error("expected statement");
                }
            }
        }
    }
    p.expect(RightCurley);

    p.close(m, Block);
}

fn stmt_let(p: &mut Parser) {
    assert!(p.at(LetKeyword));
    let m = p.open();

    p.expect(LetKeyword);
    p.expect(Identifier);
    p.expect(Equals);
    expr(p);
    p.expect(Semicolon);

    p.close(m, LetStmt);
}

fn stmt_return(p: &mut Parser) {
    assert!(p.at(ReturnKeyword));
    let m = p.open();

    p.expect(ReturnKeyword);
    expr(p);
    p.expect(Semicolon);

    p.close(m, ReturnStmt);
}

fn stmt_expr(p: &mut Parser) {
    let m = p.open();

    expr(p);
    p.expect(Semicolon);

    p.close(m, ExprStmt);
}

fn expr(p: &mut Parser) {
    expr_rec(p, Eof);
}

fn expr_rec(p: &mut Parser, left: TokenKind) {
    let Some(mut lhs) = expr_delimited(p) else {
        return;
    };

    while p.at(LeftParen) {
        let m = p.open_before(lhs);
        arg_list(p);
        lhs = p.close(m, ExprCall);
    }

    loop {
        let right = p.nth(0);
        if right_binds_tighter(left, right) {
            let m = p.open_before(lhs);
            p.advance();
            expr_rec(p, right);
            lhs = p.close(m, ExprBinary);
        } else {
            break;
        }
    }
}

fn right_binds_tighter(left: TokenKind, right: TokenKind) -> bool {
    fn tightness(kind: TokenKind) -> Option<usize> {
        [
            // Precedence table:
            [Plus, Minus].as_slice(),
            &[Star, ForwardSlash],
        ]
        .iter()
        .position(|level| level.contains(&kind))
    }
    let Some(right_tightness) = tightness(right) else {
        return false;
    };
    let Some(left_tightness) = tightness(left) else {
        assert!(left.eq(&Eof));
        return true;
    };
    right_tightness > left_tightness
}

fn expr_delimited(p: &mut Parser) -> Option<MarkClosed> {
    let result = match p.nth(0) {
        TrueKeyword | FalseKeyword | IntegerNumericValue => {
            let m = p.open();
            p.advance();
            p.close(m, ExprLiteral)
        }
        Identifier => {
            let m = p.open();
            p.advance();
            p.close(m, ExprName)
        }
        LeftParen => {
            let m = p.open();
            p.expect(LeftParen);
            expr(p);
            p.expect(RightParen);
            p.close(m, ExprParen)
        }
        _ => return None,
    };
    Some(result)
}

fn arg_list(p: &mut Parser) {
    assert!(p.at(LeftParen));
    let m = p.open();

    p.expect(LeftParen);
    while !p.at(RightParen) && !p.eof() {
        if p.at_any(EXPR_FIRST) {
            arg(p);
        } else {
            break;
        }
    }
    p.expect(RightParen);

    p.close(m, ArgList);
}

fn arg(p: &mut Parser) {
    let m = p.open();
    expr(p);
    if !p.at(RightParen) {
        p.expect(Comma);
    }
    p.close(m, Arg);
}

fn field(p: &mut Parser) {
    let m = p.open();

    p.expect(Identifier);
    p.expect(Colon);
    type_expr(p);
    p.expect(Semicolon);

    p.close(m, Field);
}

const ACCESS_MODIFIERS: &[TokenKind] = &[PublicKeyword, InternalKeyword, ProtectedKeyword];
fn access_modifier(p: &mut Parser) -> Option<MarkClosed> {
    assert!(p.at_any(ACCESS_MODIFIERS));

    let result = match p.nth(0) {
        ProtectedKeyword => {
            let m = p.open();
            p.expect(ProtectedKeyword);
            if p.at(InternalKeyword) {
                p.expect(InternalKeyword);
            }

            Some(p.close(m, AccessModifier))
        }
        PublicKeyword | InternalKeyword => {
            let m = p.open();
            p.expect(p.nth(0));
            Some(p.close(m, AccessModifier))
        }
        _ => return None,
    };
    return result;
}

#[macro_export]
macro_rules! format_to {
    ($buf:expr) => ();
    ($buf:expr, $lit:literal $($arg:tt)*) => {
        { use ::std::fmt::Write as _; let _ = ::std::write!($buf, $lit $($arg)*); }
    };
}
impl<'a> Node<'a> {
    fn print(&self, buf: &mut String, level: usize) {
        let indent = "  ".repeat(level);
        format_to!(buf, "{indent}{:?}\n", self.kind);
        for child in &self.children {
            match child {
                Child::Token(token) => {
                    format_to!(buf, "{indent}  '{}'\n", token.text)
                }
                Child::Node(node) => node.print(buf, level + 1),
            }
        }
        assert!(buf.ends_with('\n'));
    }
}

impl<'a> fmt::Debug for Node<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = String::new();
        self.print(&mut buf, 0);
        write!(f, "{}", buf)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<TokenKind, ()>;

    fn next(&mut self) -> Option<Result<TokenKind, ()>> {
        self.lexer.next()
    }
}

pub fn parse(text: &str) -> Node {
    let tokens = lex(text);
    let mut p = Parser::new(tokens);
    compilation_unit(&mut p);
    p.build_node()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let text = "
      pub class Test {
        pub _field: int;

        pub prop Value: string;
      }
    ";
        let cst = parse(text);
        println!("{cst:?}");
    }

    #[test]
    fn prop_smoke() {
        let text = "
      pub class Test {
        pub _field: int;

        prop
        pub prop Value2: string;
      }
    ";
        let cst = parse(text);
        println!("{cst:?}");
    }
}
